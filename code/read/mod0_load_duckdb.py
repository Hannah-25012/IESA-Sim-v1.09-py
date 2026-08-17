# File to build the in-memory simulation data structures directly from
# simmodel.duckdb (written by mod0_read_data_save_duck), instead of building
# them from the pandas frames read straight out of the Excel file. This is
# the "single source of truth is the database" loader: run
# mod0_read_data_save_duck once to (re)populate the database, then call
# load_from_duckdb() any number of times to get typed Struct objects back.
import duckdb
import numpy as np

from datastructures import Struct


def _pivot(df, row_col, row_order, col_col, col_order, value_col):
    """Reshape a long (row_col, col_col, value_col) table into a 2D array
    with rows/columns in exactly row_order/col_order, filling any missing
    combination with 0 (matches the .fillna(0) behaviour of the old loader).
    """
    pivot = df.pivot(index=row_col, columns=col_col, values=value_col)
    pivot = pivot.reindex(index=row_order, columns=col_order).fillna(0)
    return pivot.to_numpy()


def _pivot_carry(df, row_col, row_order, col_col, col_order, value_col):
    """Like _pivot, but a missing (row, col) combination inherits that same
    row's nearest earlier col_order entry (falling back to the nearest later
    one for a leading gap) instead of being zeroed.

    A native Sim-built database always has one technology_stocks/
    technology_costs/infrastructure_costs row per (tech, period) - every
    period is fully populated, so plain fillna(0) never actually fires. A
    merged/unified database (dbcompare-backend's /unify) can be far sparser:
    IESA-Opt itself only carries a partial "current status" snapshot for a
    near-term checkpoint year Sim never modeled (verified: a real merge had
    266 of 666 technologies - concrete existing power plants like "Electricity
    from CCGT old - Power BE" - with zero technology_stocks/technology_costs
    rows at period 2022 even though the same plants have real capacity at
    2020 and 2025). IESA-Opt.jl's own reader treats a missing (tech, period)
    key as "no constraint recorded" (see IESAOPT's input_tables.jl: a missing
    techStock_max entry means the @constraint for that (t, period) is never
    added at all) - capacity just carries over from whatever the surrounding
    periods/investment decisions already determine.  IESA-Sim's dense
    per-period arrays have no way to express "no constraint", but zeroing
    min/max/investment/fom/vom is the one value guaranteed to be wrong: a
    zeroed max clamps existing stock down to 0 in invest_techstocks_def.py
    (`min(max(tech_stock_min, tech_stock), tech_stock_max)`), forcibly
    decommissioning a still-operating plant for that single period, and a
    zeroed investment cost makes that period's (non-)investment look free.
    Carrying the nearest real value forward (back for a leading gap) is the
    closest a fixed-grid model can get to IESA-Opt's "unconstrained" reading
    without inventing a number IESA-Opt never actually provided.
    """
    pivot = df.pivot(index=row_col, columns=col_col, values=value_col)
    pivot = pivot.reindex(index=row_order, columns=col_order)
    pivot = pivot.ffill(axis=1).bfill(axis=1).fillna(0)
    return pivot.to_numpy()


def _load_parameters(con):
    # Value comes back as a Python float from a native Sim-built database
    # (Value column is DOUBLE - see mod0_read_data_save_duck), but as text
    # from a merged/unified database (the unify step matches IESA-Opt's own
    # "parameters" table, whose Value column is VARCHAR, and that table also
    # carries IESA-Opt's own non-numeric rows like scenario_description) -
    # cast just the 12 names IESA-Sim itself needs, so downstream arithmetic
    # (investment thresholds, cost comparisons) gets a float either way
    # instead of failing on `float >= str`.
    raw = dict(con.execute("SELECT Name, Value FROM parameters").fetchall())
    names = ("SPBT_benchmark", "SPBT_min", "CR_threshold", "CR_min", "NUF_threshold", "NUF_min",
             "penalization", "gas_premium", "voll", "min_spread", "gov_dr", "exports_value")
    values = {name: float(raw[name]) for name in names}
    return Struct(
        powinv=Struct(
            SPBT_benchmark=values["SPBT_benchmark"],
            SPBT_min=values["SPBT_min"],
            CR_threshold=values["CR_threshold"],
            CR_min=values["CR_min"],
            NUF_threshold=values["NUF_threshold"],
            NUF_min=values["NUF_min"],
        ),
        scarcity=Struct(
            penalization=values["penalization"],
            gas_premium=values["gas_premium"],
        ),
        voll=values["voll"],
        min_spread=values["min_spread"],
        gov_dr=values["gov_dr"],
        exports_value=values["exports_value"],
    )


def _load_types(con):
    activities_list = [r[0] for r in con.execute(
        "SELECT activity_type FROM activity_types ORDER BY seq").fetchall()]
    sectors_list = [r[0] for r in con.execute(
        "SELECT sectors FROM sectors ORDER BY seq").fetchall()]
    # labels/price_init can be legitimately different lengths (padded with NULL
    # in the table to make a rectangular frame - see dict_to_df_padded_nan), so
    # each column drops its own NULLs rather than sharing one row filter.
    labels = [r[0] for r in con.execute(
        'SELECT labels FROM energy_types WHERE labels IS NOT NULL ORDER BY seq').fetchall()]
    price_init = np.array([r[0] for r in con.execute(
        'SELECT "price init" FROM energy_types WHERE "price init" IS NOT NULL ORDER BY seq').fetchall()], dtype=float)
    return Struct(
        activities=activities_list,
        sectors=sectors_list,
        energy=Struct(labels=labels, price_init=price_init),
    )


def _load_agents(con):
    agent_type_names = [r[0] for r in con.execute(
        "SELECT Name FROM agent_types ORDER BY seq").fetchall()]

    profile_rows = con.execute(
        "SELECT Name, rates FROM agent_profiles ORDER BY seq").fetchall()
    agent_profiles = [r[0] for r in profile_rows]
    agents_dr = np.array([r[1] for r in profile_rows], dtype=float)

    multi_criteria_categories = [r[0] for r in con.execute(
        "SELECT Name FROM agent_criteria ORDER BY seq").fetchall()]

    pop_df = con.execute("SELECT agent_profile, agent_type, value FROM population").fetchdf()
    agents_populations = _pivot(pop_df, "agent_profile", agent_profiles, "agent_type", agent_type_names, "value")

    weights_df = con.execute("SELECT agent_criteria, agent_type, value FROM criteria_weights").fetchdf()
    weights_multiCriteria = _pivot(weights_df, "agent_criteria", multi_criteria_categories, "agent_type", agent_type_names, "value")

    agents = Struct(
        types=agent_type_names,
        profiles=agent_profiles,
        criteria=Struct(categories=multi_criteria_categories, weights=weights_multiCriteria),
        populations=agents_populations,
        rates=agents_dr,
    )
    return agent_type_names, agents


def _load_activities(con):
    periods = [r[0] for r in con.execute("SELECT period FROM periods ORDER BY period_order").fetchall()]

    # A native Sim-built database (mod0_read_data_save_duck) puts volumes
    # directly on "activities" as wide volumes_<year> columns; a merged/
    # unified database (dbcompare-backend's /unify) doesn't - to reconcile
    # with IESA-Opt's own long-shaped table, volumes live in a separate
    # "activity_volumes" (activity_name, period, value) table instead.
    # Detect which shape this database actually has rather than assuming.
    activities_cols = {r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'activities'"
    ).fetchall()}
    has_wide_volumes = all(f"volumes_{p}" in activities_cols for p in periods)

    cols = ['"Name"', '"activity_resolution"', '"activity_type"', '"energy_label"', '"agent_profile"']
    if has_wide_volumes:
        cols += [f'"volumes_{p}"' for p in periods]
    df = con.execute(f'SELECT {", ".join(cols)} FROM activities ORDER BY seq').fetchdf()

    names = df["Name"].tolist()
    if has_wide_volumes:
        volumes = df[[f"volumes_{p}" for p in periods]].fillna(0).to_numpy(dtype=float)
    else:
        # A merged/unified database's activity_volumes can be missing an
        # activity's row for a period IESA-Opt only partially populated
        # (same root cause as technology_stocks/technology_costs above - see
        # _pivot_carry's docstring) - zero-filling that gap would make an
        # activity's demand vanish for one period even though it has real
        # volume on both sides of the gap, so carry the nearest known volume
        # forward/back instead of assuming 0.
        av_df = con.execute("SELECT activity_name, period, value FROM activity_volumes").fetchdf()
        volumes = _pivot_carry(av_df, "activity_name", names, "period", periods, "value")

    activities = Struct(
        names=names,
        periods=np.array(periods),
        volumes=volumes,
        resolution=df["activity_resolution"].tolist(),
        types=df["activity_type"].tolist(),
        labels=df["energy_label"].tolist(),
        agents=df["agent_profile"].tolist(),
        drivers=Struct(),
        energies=Struct(),
        emissions=Struct(),
        electricity=Struct(),
        gaseous=Struct(),
        infra=Struct(),
        prices=Struct(),
    )
    return periods, names, activities


def _load_profiles(con, periods):
    profile_types = [r[0] for r in con.execute(
        "SELECT name FROM hourly_profile_types ORDER BY seq").fetchall()]
    hours = list(range(8760))

    hp_df = con.execute("SELECT hour, profile_type, value FROM hourly_profiles").fetchdf()
    shapes = _pivot(hp_df, "hour", hours, "profile_type", profile_types, "value")

    interconnectors = [r[0] for r in con.execute(
        "SELECT name FROM interconnectors ORDER BY id").fetchall()]
    nIC = len(interconnectors)

    pp_df = con.execute("SELECT hour, interconnector_id, period, price FROM price_profiles").fetchdf()
    prices = np.zeros((8760, nIC, len(periods)))
    for i in range(nIC):
        sub = pp_df[pp_df["interconnector_id"] == i]
        prices[:, i, :] = _pivot(sub, "hour", hours, "period", periods, "price")

    return Struct(types=profile_types, shapes=shapes, interconnectors=interconnectors, prices=prices)


def _table_exists(con, name):
    return con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()[0] > 0


# IESA-Opt has no 'Exports' category at all - it tags an export technology
# 'Primary' just like a genuine import/production technology, distinguishing
# it only by subsector text (e.g. "Exported Hydrogen" vs "Imported
# Hydrogen"). IESA-Sim's own native workbook, by contrast, has a real
# 'Exports' category (11 technologies: Natural Gas/naphtha/road-fuel/
# kerosene exports, "Export from NL to Power*") that
# post_primary_energy.py's own export-tracking (`if tech.category ==
# 'Exports':`) and results_system_costs.py's cost-category bucketing
# (`coord_fuel = coord_primary` - a 'Primary' tech's own VOM lands in the
# "fuels" cost bucket unconditionally) are both written assuming.
#
# A merged database's Opt-anchored 'Primary' export technology falls
# through both: it never reaches post_primary_energy.py's 'Exports' branch,
# so its physical flow never reaches results.exports (the "exporting
# revenues" system-cost line - confirmed empirically: 0.0 for "Hydrogen HD"
# in a period where the one such technology, Hyd03_02, dispatched ~1954
# units of exports), and its own VOM cost instead lands, unconditionally
# and undiluted, in "fuels" - confirmed: -49.80 EUR/unit x ~1954 units =
# -97,304 (a large NEGATIVE "fuels" cost, i.e. an unlabelled revenue
# item hiding inside what should read as a cost line).
#
# Checked IESA-Opt's own full native technology set three independent ways
# (name contains "export", subsector contains "export", and the structural
# test - a genuine export has a NEGATIVE balance on its own main activity,
# unlike every other Primary technology, which is positive) - all three
# agree Hyd03_02 is the only Primary-category technology this affects.
# "Electricity Export to EU" (PEU01_03) is the only other export-named
# technology, but it's category='XC Trade', not 'Primary' - deliberately
# left alone here: invest_techstocks_def.py's own guaranteed-availability
# check keys off `'XC Trade' in tech.category` specifically (matching how
# IESA-Opt's own solver treats import/export capacity as existing grid
# infrastructure rather than re-justifying it economically every period -
# see that file's own comment) and reclassifying it away from 'XC Trade'
# would strip that guard, re-exposing it to the same "circular
# scarcity-pricing trap" this whole investigation started from. Its own
# export volume is already correctly tracked (confirmed: results.exports
# for "Electricity NL - HV" is nonzero and grows across periods) via the
# generic activity-imbalance fallback post_primary_energy.py falls back to
# for any technology outside the 'Primary'/'Exports' categories - so
# nothing there needs fixing.
#
# Scoped to category == 'Primary' specifically (not "any category") so
# this can never touch 'XC Trade' or an already-correct native 'Exports'
# technology - only the exact gap it exists to close.
def _reclassify_opt_export_technologies(tech_df):
    is_primary = tech_df["category"] == "Primary"
    looks_like_export = (
        tech_df["name"].str.contains("export", case=False, na=False)
        | tech_df["subsector"].str.contains("export", case=False, na=False)
    )
    reclassify = is_primary & looks_like_export
    if reclassify.any():
        tech_df.loc[reclassify, "category"] = "Exports"
        print(
            f"----Reclassified {reclassify.sum()} IESA-Opt-sourced technology(ies) from "
            "'Primary' to 'Exports' (IESA-Opt has no 'Exports' category of its own): "
            f"{tech_df.loc[reclassify, 'id'].tolist()}"
        )


# IESA-Opt's own workbook carries two side-tables a merged database passes
# through verbatim (dbcompare-backend/app.py's unify() copies every opt-only
# table it doesn't otherwise merge) that a native Sim database never has:
# shed_capacity and flex_capacity, both (tech_id, period) -> already-derived
# per-unit-capacity ceiling (IESA-Opt's own compute_shed_capacity!/
# compute_flex_capacity!: raw_percentage * peak(hourly_profile) * cap2act -
# see julia-backend/src/parameters.jl). Absent entirely from a native Sim
# database (no read_data_save_duck writer ever creates them), so every call
# site here is guarded by _table_exists and a no-op when they're missing.
#
# The two get treated differently because IESA-Sim's own dispatch treats
# their raw technologies-table counterpart differently:
#   - shedding_capacity: disp_initialize_power.py already re-derives the
#     same ceiling itself (raw_percentage * peak * cap2act, fixed by "Match
#     IESA-Opt's shed_capacity derivation" 2026-08-10) - re-deriving on top
#     of shed_capacity's own already-derived value here would double-apply
#     the peak/cap2act scaling and silently shrink it. So this table is used
#     for VALIDATION only: recompute Sim's own ceiling from the raw column
#     and compare, to catch a future drift between the two derivations
#     instead of silently trusting they still agree.
#   - flexibility_capacity: disp_initialize_power.py/disp_power_batteries.py/
#     disp_power_loadshifting.py use it directly as a final per-hour
#     charge/discharge (or shift) RATE with no derivation step at all - so
#     for a technology carrying IESA-Opt's raw, undereived percentage (its
#     technologies.flexibility_capacity is 1.0 for e.g. every merged NL
#     battery technology, meant to be scaled the same way shedding_capacity
#     is, but never was), that raw value is silently wrong by orders of
#     magnitude (verified: PNL03_03 "Battery Storage Daily" carries a raw
#     1.0 - a full-stock cycle every hour - when IESA-Opt's own derived
#     value is 0.0036). This one IS overridden with flex_capacity's
#     already-correct, already-derived value wherever a row exists.
def _apply_opt_flex_shed_tables(con, tech_df):
    if _table_exists(con, "shed_capacity"):
        # Validate only - see the derivation-ownership note above. Compares
        # Sim's own re-derived ceiling (raw shedding_capacity * profile peak
        # * cap2act) against IESA-Opt's own already-derived shed_capacity
        # value for the same (tech, period) and flags any tech where they
        # disagree by more than a small floating-point tolerance, so a
        # future change to either derivation doesn't silently drift apart.
        mismatch = con.execute("""
            SELECT sc.key1 AS tech_id, sc.key2 AS period, sc.value AS opt_value,
                   t.shedding_capacity * pk.peak * t.cap2act AS sim_value
            FROM shed_capacity sc
            JOIN technologies t ON t.id = sc.key1
            JOIN (SELECT profile_type, max(value) AS peak FROM hourly_profiles GROUP BY profile_type) pk
                ON pk.profile_type = t.hourly_profile
            WHERE abs(t.shedding_capacity * pk.peak * t.cap2act - sc.value)
                  > 0.02 * abs(sc.value) + 1e-9
        """).fetchdf()
        if not mismatch.empty:
            print(
                f"!!!! Warning: {len(mismatch)} (tech, period) row(s) where IESA-Sim's own "
                "re-derived shedding ceiling disagrees with IESA-Opt's shed_capacity table "
                "by more than 2% - shedding_capacity's derivation may have drifted:"
            )
            print(mismatch.to_string(index=False))

    if _table_exists(con, "flex_capacity"):
        # A tech's flex_capacity value is constant across every period it
        # appears in for all data seen so far (both raw_percentage and
        # cap2act are themselves period-invariant technology attributes, and
        # profile peak doesn't vary by solve period either) - IESA-Sim has no
        # period dimension for this field to represent per-period variation
        # even if a future dataset had it, so collapse to one value per tech
        # by averaging, and flag it loudly if that average is hiding real
        # spread rather than silently rounding it away.
        fc = con.execute("SELECT key1 AS tech_id, key2 AS period, value FROM flex_capacity").fetchdf()
        spread = fc.groupby("tech_id")["value"].agg(["mean", "std"])
        noisy = spread[spread["std"].fillna(0) > 0.02 * spread["mean"].abs()]
        if not noisy.empty:
            print(
                f"!!!! Warning: {len(noisy)} technology(ies) have period-varying flex_capacity "
                "IESA-Sim collapses to a single mean value (no period dimension for this field):"
            )
            print(noisy.to_string())
        overrides = spread["mean"].to_dict()
        applied = tech_df["id"].isin(overrides)
        if applied.any():
            tech_df.loc[applied, "flexibility_capacity"] = tech_df.loc[applied, "id"].map(overrides)
            print(
                f"----Applied IESA-Opt's own derived flex_capacity to {applied.sum()} "
                "technology(ies) whose raw flexibility_capacity is an undereived percentage."
            )


def _load_technologies(con, periods, activities_names):
    tech_cols = [
        "id", "category", "sector", "subsector", "name", "unit", "activity", "cap2act", "lifetime",
        "dispatch_type", "hourly_profile", "social_perception", "perceived_complexity",
        "subsidy_subject", "feedin_subject", "shedding_capacity", "shedding_limits", "shedding_guarantee",
        "flexibility_form", "flexibility_capacity", "flexibility_volume", "flexibility_range",
        "flexibility_losses", "flexibility_nonnegotiable", "buffer_up", "buffer_down", "buffer_capacity",
        "stock_deploy", "stock_initial",
    ]
    # wacc only exists on a merged/unified database (dbcompare-backend's
    # /unify carries IESA-Opt's own per-technology wacc through; a native
    # Sim-built database - mod0_read_data_save_duck - has no such column,
    # since IESA-Sim historically prices all technologies via their agent's
    # discount rate instead - see mod1_initialize's annuity factor loop).
    tech_table_cols = {r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'technologies'"
    ).fetchall()}
    has_wacc = "wacc" in tech_table_cols
    if has_wacc:
        tech_cols.append("wacc")
    quoted = ", ".join(f'"{c}"' for c in tech_cols)
    tech_df = con.execute(f'SELECT {quoted} FROM technologies ORDER BY seq').fetchdf()
    if not has_wacc:
        tech_df["wacc"] = np.nan
    _reclassify_opt_export_technologies(tech_df)
    _apply_opt_flex_shed_tables(con, tech_df)
    tech_balancers = tech_df["id"].tolist()

    # Positionally aligned with tech_balancers (None where a tech has no flex
    # coupling), not compacted, so it can be zipped/indexed by technology
    # position - e.g. for resolving each Technology's flexibility.activity FK.
    flex_act_df = con.execute("SELECT tech_id, activity_name FROM technology_flexibility_activities").fetchdf()
    flex_act_map = dict(zip(flex_act_df["tech_id"], flex_act_df["activity_name"]))
    flexibility_activity = [flex_act_map.get(t) for t in tech_balancers]

    costs_df = con.execute("SELECT tech_id, period, investment, fom, vom FROM technology_costs").fetchdf()
    inv_cost = _pivot_carry(costs_df, "tech_id", tech_balancers, "period", periods, "investment")
    fom_cost = _pivot_carry(costs_df, "tech_id", tech_balancers, "period", periods, "fom")
    vom_cost = _pivot_carry(costs_df, "tech_id", tech_balancers, "period", periods, "vom")

    stocks_df = con.execute("SELECT tech_id, period, dec_planned, min, max FROM technology_stocks").fetchdf()
    tech_stock_dec = _pivot(stocks_df, "tech_id", tech_balancers, "period", periods, "dec_planned")
    tech_stock_min = _pivot_carry(stocks_df, "tech_id", tech_balancers, "period", periods, "min")
    tech_stock_max = _pivot_carry(stocks_df, "tech_id", tech_balancers, "period", periods, "max")

    # A merged/unified database's energy_balance table can legitimately repeat
    # the same (tech_id, activity_name) key once per period (see
    # dbcompare-backend/app.py's _IESA_OPT_SIM_SHARED_TABLES entry for this
    # table) - IESA-Sim has no period dimension for these coefficients, so
    # collapse to one row per key, but verify the repeats actually agree
    # rather than silently picking one if a real conflict ever shows up.
    eb_df = con.execute("SELECT tech_id, activity_name, value FROM energy_balance").fetchdf()
    value_counts = eb_df.groupby(["tech_id", "activity_name"])["value"].nunique()
    conflicts = value_counts[value_counts > 1]
    if not conflicts.empty:
        raise ValueError(
            "energy_balance has period-varying values for "
            f"{len(conflicts)} (tech_id, activity_name) pair(s) that IESA-Sim "
            f"cannot represent (no period dimension), e.g. {conflicts.index[0]}"
        )
    eb_df = eb_df.drop_duplicates(subset=["tech_id", "activity_name"])
    activity_balances = _pivot(eb_df, "tech_id", tech_balancers, "activity_name", activities_names, "value")

    balancers = Struct(
        ids=tech_balancers,
        names=tech_df["name"].tolist(),
        sectors=tech_df["sector"].tolist(),
        subsectors=tech_df["subsector"].tolist(),
        units=tech_df["unit"].tolist(),
        activities=tech_df["activity"].tolist(),
        categories=tech_df["category"].tolist(),
        costs=Struct(
            investments=inv_cost, foms=fom_cost, voms=vom_cost,
            lifetimes=tech_df["lifetime"].to_numpy(dtype=float),
            wacc=tech_df["wacc"].to_numpy(dtype=float),
        ),
        cap2acts=tech_df["cap2act"].to_numpy(dtype=float),
        dispatch=tech_df["dispatch_type"].tolist(),
        profiles=tech_df["hourly_profile"].tolist(),
        agents=Struct(
            social_perception=tech_df["social_perception"].tolist(),
            complexity=tech_df["perceived_complexity"].tolist(),
        ),
        policies=Struct(
            # NULL for technologies that came from IESA-Opt (see
            # dbcompare-backend/app.py's technologies entry - these two
            # columns are IESA-Sim-only concepts with no IESA-Opt
            # counterpart) - treat as not subsidy/feedin-subject.
            subsidy_subject=tech_df["subsidy_subject"].fillna(False).to_numpy(dtype=bool),
            feedin_subject=tech_df["feedin_subject"].fillna(False).to_numpy(dtype=bool),
        ),
        shedding=Struct(
            capacity=tech_df["shedding_capacity"].to_numpy(dtype=float),
            limits=tech_df["shedding_limits"].to_numpy(dtype=float),
            guarantee=tech_df["shedding_guarantee"].to_numpy(dtype=float),
        ),
        flexibility=Struct(
            form=tech_df["flexibility_form"].tolist(),
            activity=flexibility_activity,
            capacity=tech_df["flexibility_capacity"].to_numpy(dtype=float),
            volume=tech_df["flexibility_volume"].to_numpy(dtype=float),
            range=tech_df["flexibility_range"].tolist(),
            losses=tech_df["flexibility_losses"].to_numpy(dtype=float),
            nonnegotiable=tech_df["flexibility_nonnegotiable"].to_numpy(dtype=float),
        ),
        buffers=Struct(
            up=tech_df["buffer_up"].to_numpy(dtype=float),
            down=tech_df["buffer_down"].to_numpy(dtype=float),
            capacity=tech_df["buffer_capacity"].to_numpy(dtype=float),
        ),
        stocks=Struct(
            deploy=tech_df["stock_deploy"].to_numpy(dtype=float),
            initial=tech_df["stock_initial"].to_numpy(dtype=float),
            dec_planned=tech_stock_dec, min=tech_stock_min, max=tech_stock_max,
        ),
        drivers=Struct(), energies=Struct(), emissions=Struct(), use=Struct(),
        investments=Struct(), retrofittings=Struct(), decommissionings=Struct(),
        generators=Struct(), loops=Struct(), mca=Struct(),
        activity_balances=activity_balances,
    )

    infra_cols = ["id", "category", "name", "unit", "activity", "cap2act", "lifetime", "stock_initial"]
    quoted = ", ".join(f'"{c}"' for c in infra_cols)
    infra_df = con.execute(f'SELECT {quoted} FROM infrastructure ORDER BY seq').fetchdf()
    tech_infra = infra_df["id"].tolist()

    infra_costs_df = con.execute("SELECT infra_id, period, investment, fom FROM infrastructure_costs").fetchdf()
    inv_cost_infra = _pivot_carry(infra_costs_df, "infra_id", tech_infra, "period", periods, "investment")
    fom_cost_infra = _pivot_carry(infra_costs_df, "infra_id", tech_infra, "period", periods, "fom")

    infra = Struct(
        ids=tech_infra,
        categories=infra_df["category"].tolist(),
        names=infra_df["name"].tolist(),
        units=infra_df["unit"].tolist(),
        activity=infra_df["activity"].tolist(),
        costs=Struct(
            investments=inv_cost_infra, foms=fom_cost_infra,
            lifetimes=infra_df["lifetime"].to_numpy(dtype=float),
        ),
        cap2acts=infra_df["cap2act"].to_numpy(dtype=float),
        stocks=Struct(initial=infra_df["stock_initial"].to_numpy(dtype=float)),
    )

    # Same period-repetition as energy_balance above (see
    # dbcompare-backend/app.py's _IESA_OPT_SIM_SHARED_TABLES entry for
    # retrofittings) - collapse to one row per (from_tech, to_tech), erroring
    # if the repeats ever actually disagree on cost.
    retro_df = con.execute("SELECT from_tech, to_tech, cost FROM retrofittings").fetchdf()
    retro_conflicts = retro_df.groupby(["from_tech", "to_tech"])["cost"].nunique()
    retro_conflicts = retro_conflicts[retro_conflicts > 1]
    if not retro_conflicts.empty:
        raise ValueError(
            "retrofittings has period-varying costs for "
            f"{len(retro_conflicts)} (from_tech, to_tech) pair(s) that IESA-Sim "
            f"cannot represent (no period dimension), e.g. {retro_conflicts.index[0]}"
        )
    retro_df = retro_df.drop_duplicates(subset=["from_tech", "to_tech"])
    retrofittings = Struct(**{
        "to": retro_df["to_tech"].tolist(),
        "from": retro_df["from_tech"].tolist(),
        "costs": retro_df["cost"].tolist(),
    })

    return Struct(balancers=balancers, infra=infra, retrofittings=retrofittings)


def _load_node_emission_targets(con, periods):
    """IESA-Opt-only policy input (not part of dbcompare-backend/app.py's
    _IESA_OPT_SIM_SHARED_TABLES, so a native Sim-built database never has
    it, and a merged/unified database only carries it through when
    IESA-Opt itself provided targets) - per-node, per-period emission caps.
    Node-keyed rather than activity-keyed, so it doesn't fit _load_policies'
    own load_block helper below. Returns an empty Struct (nodes=[]) when
    the table doesn't exist, so callers can branch on `"X" in nodes` rather
    than needing their own try/except."""
    table_exists = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'node_emission_targets'"
    ).fetchone()[0] > 0
    if not table_exists:
        return Struct(nodes=[])
    df = con.execute(
        "SELECT node, period, target_air, target_all, target_bunker, target_feedstock "
        "FROM node_emission_targets"
    ).fetchdf()
    nodes = sorted(df["node"].unique().tolist())
    return Struct(
        nodes=nodes,
        periods=np.array(periods),
        air=_pivot(df, "node", nodes, "period", periods, "target_air"),
        all=_pivot(df, "node", nodes, "period", periods, "target_all"),
        bunker=_pivot(df, "node", nodes, "period", periods, "target_bunker"),
        feedstock=_pivot(df, "node", nodes, "period", periods, "target_feedstock"),
    )


def _load_policies(con, periods):
    def load_block(table):
        df = con.execute(f"SELECT activity_name, period, value, seq FROM {table}").fetchdf()
        order = df[["activity_name", "seq"]].drop_duplicates().sort_values("seq")
        act_names = order["activity_name"].tolist()
        values = _pivot(df, "activity_name", act_names, "period", periods, "value")
        return Struct(activities=act_names, values=values)

    return Struct(
        taxes=load_block("policy_taxes"),
        feedins=load_block("policy_feedins"),
        subsidies=load_block("policy_subsidies"),
        emission_targets=_load_node_emission_targets(con, periods),
    )


def load_from_duckdb(db_path="SIMmodel.duckdb"):
    con = duckdb.connect(db_path, read_only=True)
    try:
        parameters = _load_parameters(con)
        types_data = _load_types(con)
        agent_type_names, agents = _load_agents(con)
        periods, activities_names, activities = _load_activities(con)
        profiles = _load_profiles(con, periods)
        technologies = _load_technologies(con, periods, activities_names)
        policies = _load_policies(con, periods)
    finally:
        con.close()

    return parameters, types_data, activities, profiles, technologies, agents, policies
