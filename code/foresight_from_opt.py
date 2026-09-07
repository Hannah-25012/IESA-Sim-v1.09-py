"""
Opt-first perfect-foresight price feed for IESA-Sim.

IESA-Opt solves its whole horizon jointly in one LP by default (periods_solve
defaults to every period - julia-backend/src/data_reading.jl; solve.jl builds
one Model and calls optimize! once) - genuine perfect foresight, unlike Sim's
own self-iterated foresight (main.py's multi-pass Picard iteration, which only
ever converges toward what Sim itself would have realized). This module reads
an already-solved Opt results DuckDB and builds a foresight_reference Struct
with the SAME shape main.py's self-iteration produces
(.energy_prices [nAe,nP], .emission_prices [nAc,nP]) - see
invest/invest_tech_LCOPs.py's docstring for how that Struct is consumed. No
change was needed there: this only replaces the *producer* of the reference.

Pure duckdb + numpy, no IESA-Sim project imports except datastructures.Struct
(a plain SimpleNamespace) - mirrors handoff_tier1.py's style so this stays
importable from a bare script if ever needed.
"""
import os

import duckdb
import numpy as np

from datastructures import Struct

# Sim's own activities.emissions.targets (mod1_initialize.py:189) only names
# the two ETS/n-ETS activities; Opt's emission_prices table instead carries
# shadow prices for its 3 aggregate policy caps (emTargetAir/Bunker/FS -
# julia-backend/src/writers.jl:611-616). This is the same activity<->cap
# correspondence dbcompare-backend's results-comparison dashboard hardcodes
# for its own (offline, non-runtime) SQL - see commit 9f68b8e - ported here
# since no general alias table exists for it.
EMISSION_CAP_TO_ACTIVITIES = {
    "emTargetAir": ("CO2 Air ETS", "CO2 Air n-ETS"),
    "emTargetBunker": ("CO2 Air Int. Transport",),
    "emTargetFS": ("CO2 Air Feedstock end of lifetime",),
}


def load_opt_price_reference(opt_results_db, periods, energy_names, emission_names,
                              fallback_energy, fallback_emission):
    """Build (reference, report). `reference` is a Struct ready to pass as
    main.py's foresight_reference. `report` is a plain dict of coverage
    counts - how much of the reference is actually Opt-sourced vs held flat
    at the fallback, since silently mixing the two would be hard to notice
    otherwise.

    periods: Sim's own real-year period vector (activities.periods).
    energy_names / emission_names: Sim's own activity name lists, in the same
    row order invest_tech_LCOPs.py indexes with (activities.energies.names /
    activities.emissions.names - both built in mod1_initialize.py by
    filtering activities.entities for is_energy/is_emission in the entities'
    original order, the same filter/order post_prices.py and
    invest_tech_LCOPs.py themselves use for `energy_activities`).
    fallback_energy / fallback_emission: flat per-row value to use for any
    (row, period) Opt doesn't cover, broadcast across every period -
    normally activities.energies/emissions.prices.initialized, so an
    Opt-uncovered row contributes no foresight signal instead of an
    implicit free-commodity bias.

    Raises ValueError on a missing file/table (user-actionable).
    """
    if not os.path.isfile(opt_results_db):
        raise ValueError(f"not found: {opt_results_db}")

    nP = len(periods)
    nAe = len(energy_names)
    nAc = len(emission_names)
    energy_prices = np.tile(
        np.asarray(fallback_energy, dtype=float).reshape(nAe, 1), (1, nP))
    emission_prices = np.tile(
        np.asarray(fallback_emission, dtype=float).reshape(nAc, 1), (1, nP))

    period_idx = {p: i for i, p in enumerate(periods)}
    energy_idx = {name: i for i, name in enumerate(energy_names)}
    emission_idx = {name: i for i, name in enumerate(emission_names)}

    conflicts = set()
    energy_matched_names = set()
    energy_matched_cells = 0
    emission_matched_names = set()
    emission_matched_cells = 0

    con = duckdb.connect(opt_results_db, read_only=True)
    try:
        tables = {r[0] for r in con.execute("show tables").fetchall()}
        if "activity_prices" not in tables:
            raise ValueError(
                f"{opt_results_db} has no activity_prices table - was this "
                f"produced by a completed Opt solve?")

        # --- energy prices: direct name join (Opt's `activity` string is
        # Sim's activity name - same namespace, no bridge table needed, same
        # join handoff_tier1.py:280-281 already relies on for the single-leg
        # price feed). One activity can appear under several constraint
        # kinds (balance/balanceFix/balanceMatconv - writers.jl:581-606);
        # prefer 'balance' and flag any (activity, period) where the kinds'
        # prices actually disagree. -----------------------------------------
        by_key = {}
        for activity, period, price, kind in con.execute(
                # "constraint" is a reserved word in DuckDB's SQL dialect -
                # must be quoted even though writers.jl (writers.jl:606)
                # names the actual column that unquoted.
                'select activity, period, price, "constraint" from activity_prices').fetchall():
            key = (activity, period)
            if key in by_key:
                prev_kind, prev_price = by_key[key]
                if abs(prev_price - price) > 1e-9:
                    conflicts.add(key)
                if kind != "balance" and prev_kind == "balance":
                    continue
            by_key[key] = (kind, price)

        for (activity, period), (_kind, price) in by_key.items():
            iAe = energy_idx.get(activity)
            iP = period_idx.get(period)
            if iAe is None or iP is None:
                continue
            energy_prices[iAe, iP] = price
            energy_matched_names.add(activity)
            energy_matched_cells += 1

        # --- emission prices: fan the 3 aggregate caps out onto every Sim
        # activity that cap covers. Prefer node='NL' when present (this is a
        # single-node NL model in practice), else average across nodes. -----
        if "emission_prices" in tables:
            by_cap_period = {}
            for name, node, period, price in con.execute("""
                    select name, node, period, price from emission_prices
                    where name in ('emTargetAir', 'emTargetBunker', 'emTargetFS')
                    """).fetchall():
                by_cap_period.setdefault((name, period), []).append((node, price))

            cap_price = {}
            for (name, period), vals in by_cap_period.items():
                nl = [p for n, p in vals if n == "NL"]
                cap_price[(name, period)] = nl[0] if nl else float(np.mean([p for _, p in vals]))

            for cap_name, activities_for_cap in EMISSION_CAP_TO_ACTIVITIES.items():
                for activity in activities_for_cap:
                    iAc = emission_idx.get(activity)
                    if iAc is None:
                        continue
                    covered_any = False
                    for period in periods:
                        price = cap_price.get((cap_name, period))
                        if price is None:
                            continue
                        emission_prices[iAc, period_idx[period]] = price
                        emission_matched_cells += 1
                        covered_any = True
                    if covered_any:
                        emission_matched_names.add(activity)
    finally:
        con.close()

    report = {
        "energy_activities_total": nAe,
        "energy_activities_opt_covered": len(energy_matched_names),
        "energy_cells_opt_covered": energy_matched_cells,
        "energy_constraint_kind_conflicts": sorted(conflicts),
        "emission_activities_total": nAc,
        "emission_activities_opt_covered": len(emission_matched_names),
        "emission_cells_opt_covered": emission_matched_cells,
    }
    reference = Struct(energy_prices=energy_prices, emission_prices=emission_prices)
    return reference, report
