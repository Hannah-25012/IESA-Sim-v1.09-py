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


def _load_parameters(con):
    powinv = dict(con.execute("SELECT Name, Value FROM powinv").fetchall())
    scarcity = dict(con.execute("SELECT Name, Value FROM scarcity").fetchall())
    short = dict(con.execute("SELECT Name, Value FROM original_params_short").fetchall())
    return Struct(
        powinv=Struct(
            SPBT_benchmark=powinv["SPBT_benchmark"],
            SPBT_min=powinv["SPBT_min"],
            CR_threshold=powinv["CR_threshold"],
            CR_min=powinv["CR_min"],
            NUF_threshold=powinv["NUF_threshold"],
            NUF_min=powinv["NUF_min"],
        ),
        scarcity=Struct(
            penalization=scarcity["penalization"],
            gas_premium=scarcity["gas_premium"],
        ),
        voll=short["voll"],
        min_spread=short["min_spread"],
        gov_dr=short["gov_dr"],
        exports_value=short["exports_value"],
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

    volume_cols = [f'"volumes_{p}"' for p in periods]
    cols = ['"Name"', '"activity_resolution"', '"activity_type"', '"energy_label"', '"agent_profile"'] + volume_cols
    df = con.execute(f'SELECT {", ".join(cols)} FROM activities ORDER BY seq').fetchdf()

    names = df["Name"].tolist()
    volumes = df[[f"volumes_{p}" for p in periods]].fillna(0).to_numpy(dtype=float)

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


def _load_technologies(con, periods, activities_names):
    tech_cols = [
        "id", "category", "sector", "subsector", "name", "unit", "activity", "cap2act", "lifetime",
        "dispatch_type", "hourly_profile", "social_perception", "perceived_complexity",
        "subsidy_subject", "feedin_subject", "shedding_capacity", "shedding_limits", "shedding_guarantee",
        "flexibility_form", "flexibility_capacity", "flexibility_volume", "flexibility_range",
        "flexibility_losses", "flexibility_nonnegotiable", "buffer_up", "buffer_down", "buffer_capacity",
        "stock_deploy", "stock_initial",
    ]
    quoted = ", ".join(f'"{c}"' for c in tech_cols)
    tech_df = con.execute(f'SELECT {quoted} FROM technologies ORDER BY seq').fetchdf()
    tech_balancers = tech_df["id"].tolist()

    flex_act_df = con.execute("SELECT tech_id, activity_name FROM technology_flexibility_activities").fetchdf()
    flex_act_map = dict(zip(flex_act_df["tech_id"], flex_act_df["activity_name"]))
    flexibility_activity = [flex_act_map[t] for t in tech_balancers if t in flex_act_map]

    costs_df = con.execute("SELECT tech_id, period, investment, fom, vom FROM technology_costs").fetchdf()
    inv_cost = _pivot(costs_df, "tech_id", tech_balancers, "period", periods, "investment")
    fom_cost = _pivot(costs_df, "tech_id", tech_balancers, "period", periods, "fom")
    vom_cost = _pivot(costs_df, "tech_id", tech_balancers, "period", periods, "vom")

    stocks_df = con.execute("SELECT tech_id, period, dec_planned, min, max FROM technology_stocks").fetchdf()
    tech_stock_dec = _pivot(stocks_df, "tech_id", tech_balancers, "period", periods, "dec_planned")
    tech_stock_min = _pivot(stocks_df, "tech_id", tech_balancers, "period", periods, "min")
    tech_stock_max = _pivot(stocks_df, "tech_id", tech_balancers, "period", periods, "max")

    eb_df = con.execute("SELECT tech_id, activity_name, value FROM energy_balance").fetchdf()
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
        ),
        cap2acts=tech_df["cap2act"].to_numpy(dtype=float),
        dispatch=tech_df["dispatch_type"].tolist(),
        profiles=tech_df["hourly_profile"].tolist(),
        agents=Struct(
            social_perception=tech_df["social_perception"].tolist(),
            complexity=tech_df["perceived_complexity"].tolist(),
        ),
        policies=Struct(
            subsidy_subject=tech_df["subsidy_subject"].to_numpy(dtype=bool),
            feedin_subject=tech_df["feedin_subject"].to_numpy(dtype=bool),
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
    inv_cost_infra = _pivot(infra_costs_df, "infra_id", tech_infra, "period", periods, "investment")
    fom_cost_infra = _pivot(infra_costs_df, "infra_id", tech_infra, "period", periods, "fom")

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

    retro_df = con.execute("SELECT from_tech, to_tech, cost FROM retrofittings").fetchdf()
    retrofittings = Struct(**{
        "to": retro_df["to_tech"].tolist(),
        "from": retro_df["from_tech"].tolist(),
        "costs": retro_df["cost"].tolist(),
    })

    return Struct(balancers=balancers, infra=infra, retrofittings=retrofittings)


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
