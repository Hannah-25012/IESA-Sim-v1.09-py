# File to persist simulation output into relational duckDB databases.
#
# Two databases are produced, both using the same long-format-with-foreign-keys
# convention as mod0_read_data_save_duck.py's input tables (dimension tables for
# activities/technologies/periods/categories, fact tables referencing them by
# id/name instead of repeating descriptive columns on every row):
#
# - save_state_duckdb(): the full end-of-run state - the same scope as the
#   simulation_results.pkl dump (dimensions/parameters/types/activities/
#   technologies/profiles/results).
# - save_excel_duckdb(): exactly what results_write.py writes to the Excel
#   reports (system costs, emissions, sectoral balance, policy cashflows,
#   tech use/stock/investments, LCOPs, MCAs, agent choices, energy/emission
#   prices, hourly power/gas prices), as relational tables instead of
#   spreadsheet cells.
import os
import duckdb
import numpy as np
import pandas as pd


def _sql_type(dtype):
    if "bool" in str(dtype):
        return "BOOLEAN"
    elif "int" in str(dtype):
        return "INTEGER"
    elif "float" in str(dtype):
        return "DOUBLE"
    else:
        return "VARCHAR"


def _create_table_sql(table_name, df, pk=None, fks=None):
    cols_sql = ",\n    ".join(f'"{col}" {_sql_type(dt)}' for col, dt in df.dtypes.items())
    constraints = []
    if pk:
        pk_cols = pk if isinstance(pk, str) else ", ".join(pk)
        constraints.append(f"PRIMARY KEY ({pk_cols})")
    for col, ref_table, ref_col in (fks or []):
        constraints.append(f'FOREIGN KEY ("{col}") REFERENCES {ref_table}({ref_col})')
    parts = [cols_sql] + constraints
    return f"CREATE TABLE {table_name} (\n    " + ",\n    ".join(parts) + "\n)"


def _save_table(con, table_name, df, pk=None, fks=None):
    con.execute(_create_table_sql(table_name, df, pk=pk, fks=fks))
    con.execute(f"INSERT INTO {table_name} SELECT * FROM df")


def _long2(row_ids, row_col, col_ids, col_col, matrix, value_col):
    """2D (row x col) array -> long DataFrame."""
    idx = pd.MultiIndex.from_product([row_ids, col_ids], names=[row_col, col_col])
    return pd.DataFrame({value_col: np.asarray(matrix).reshape(-1)}, index=idx).reset_index()


def _long3(row_ids, row_col, mid_ids, mid_col, page_ids, page_col, matrix, value_col):
    """3D (row x mid x page) array -> long DataFrame."""
    idx = pd.MultiIndex.from_product([row_ids, mid_ids, page_ids], names=[row_col, mid_col, page_col])
    return pd.DataFrame({value_col: np.asarray(matrix).reshape(-1)}, index=idx).reset_index()


def _periods_df(periods):
    return pd.DataFrame({"period": list(periods), "period_order": range(len(periods))})


def _activities_dim_df(activity_entities):
    return pd.DataFrame({
        "name": [a.name for a in activity_entities],
        "seq": [a.idx for a in activity_entities],
        "resolution": [a.resolution for a in activity_entities],
        "type": [a.type for a in activity_entities],
        "label": [a.label for a in activity_entities],
        "agent_profile": [a.agent_profile_name for a in activity_entities],
    })


def _technologies_dim_df(tech_entities):
    return pd.DataFrame({
        "id": [t.id for t in tech_entities],
        "seq": [t.idx for t in tech_entities],
        "category": [t.category for t in tech_entities],
        "sector": [t.sector for t in tech_entities],
        "subsector": [t.subsector for t in tech_entities],
        "name": [t.name for t in tech_entities],
        "unit": [t.unit for t in tech_entities],
        "activity_name": [t.activity_name for t in tech_entities],
        "cap2act": [t.cap2act for t in tech_entities],
        "lifetime": [t.lifetime for t in tech_entities],
        "dispatch": [t.dispatch for t in tech_entities],
        "profile": [t.profile for t in tech_entities],
        "social_perception": [t.social_perception for t in tech_entities],
        "complexity": [t.complexity for t in tech_entities],
        "subsidy_subject": [bool(t.subsidy_subject) for t in tech_entities],
        "feedin_subject": [bool(t.feedin_subject) for t in tech_entities],
        "shedding_capacity": [t.shedding_capacity for t in tech_entities],
        "shedding_limits": [t.shedding_limits for t in tech_entities],
        "shedding_guarantee": [t.shedding_guarantee for t in tech_entities],
        "flexibility_form": [t.flexibility_form for t in tech_entities],
        "flexibility_activity_name": [t.flexibility_activity_name for t in tech_entities],
        "flexibility_capacity": [t.flexibility_capacity for t in tech_entities],
        "flexibility_volume": [t.flexibility_volume for t in tech_entities],
        "flexibility_range": [t.flexibility_range for t in tech_entities],
        "flexibility_losses": [t.flexibility_losses for t in tech_entities],
        "flexibility_nonnegotiable": [t.flexibility_nonnegotiable for t in tech_entities],
        "buffer_up": [t.buffer_up for t in tech_entities],
        "buffer_down": [t.buffer_down for t in tech_entities],
        "buffer_capacity": [t.buffer_capacity for t in tech_entities],
        "stock_deploy": [t.stock_deploy for t in tech_entities],
        "stock_initial": [t.stock_initial for t in tech_entities],
    })


def _dim_lookup_df(names):
    return pd.DataFrame({"idx": range(len(names)), "name": list(names)})


def save_state_duckdb(dimensions, parameters, types, activities, technologies, profiles, results, agents, db_path):
    """Persist the full end-of-run state to a relational duckDB database.

    Covers the same scope as the simulation_results.pkl dump (dimensions,
    parameters, types, activities, technologies, profiles, results).
    `agents` isn't part of that scope but is accepted purely to label the
    MCA-category and agent-type dimensions instead of leaving them as bare
    integer indices.
    """
    if os.path.exists(db_path):
        os.remove(db_path)
    con = duckdb.connect(db_path)

    nH = dimensions['nH']
    hours = list(range(nH))
    periods = list(activities.periods)
    activity_names = list(activities.names)
    activity_entities = activities.entities
    tech_entities = technologies.balancers.entities
    tech_ids = [t.id for t in tech_entities]
    infra_entities = technologies.infra.entities
    infra_ids = [i.id for i in infra_entities]
    energy_activities = [a for a in activity_entities if a.is_energy]
    energy_names = [a.name for a in energy_activities]
    emission_activities = [a for a in activity_entities if a.is_emission]
    emission_names = [a.name for a in emission_activities]

    # === dimensions (single row) ===
    _save_table(con, "dimensions", pd.DataFrame([dict(dimensions.items())]))

    # === parameters (flattened to name/value) ===
    param_rows = [
        ("powinv_SPBT_benchmark", parameters.powinv.SPBT_benchmark),
        ("powinv_SPBT_min", parameters.powinv.SPBT_min),
        ("powinv_CR_threshold", parameters.powinv.CR_threshold),
        ("powinv_CR_min", parameters.powinv.CR_min),
        ("powinv_NUF_threshold", parameters.powinv.NUF_threshold),
        ("powinv_NUF_min", parameters.powinv.NUF_min),
        ("scarcity_penalization", parameters.scarcity.penalization),
        ("scarcity_gas_premium", parameters.scarcity.gas_premium),
        ("voll", parameters.voll),
        ("min_spread", parameters.min_spread),
        ("gov_dr", parameters.gov_dr),
        ("exports_value", parameters.exports_value),
    ]
    _save_table(con, "parameters", pd.DataFrame(param_rows, columns=["name", "value"]), pk="name")

    # === periods ===
    _save_table(con, "periods", _periods_df(periods), pk="period")

    # === types ===
    _save_table(con, "type_categories", pd.DataFrame({"category": types.activities, "seq": range(len(types.activities))}), pk="category")
    _save_table(con, "type_sectors", pd.DataFrame({"sector": types.sectors, "seq": range(len(types.sectors))}), pk="sector")
    # labels/price_init can be legitimately different lengths (see
    # mod0_load_duckdb._load_types), so price_init is padded with NaN rather
    # than assumed to line up 1:1 with every label.
    price_init_padded = list(types.energy.price_init) + [np.nan] * (len(types.energy.labels) - len(types.energy.price_init))
    _save_table(con, "type_energy_labels", pd.DataFrame({
        "label": types.energy.labels, "seq": range(len(types.energy.labels)), "price_init": price_init_padded,
    }), pk="label")
    _save_table(con, "type_policy_cashflow_categories", pd.DataFrame({
        "category": types.policy_cashflows_categories, "seq": range(len(types.policy_cashflows_categories)),
    }), pk="category")

    # === lookup/dimension tables for otherwise-unnamed matrix axes ===
    _save_table(con, "lcop_categories", _dim_lookup_df(technologies.balancers.lcops.categories), pk="idx")
    _save_table(con, "mca_categories", _dim_lookup_df(agents.criteria.categories), pk="idx")
    _save_table(con, "agent_types", _dim_lookup_df(agents.types), pk="idx")
    _save_table(con, "cost_categories", _dim_lookup_df(results.costs.categories), pk="idx")
    price_range_pcts = list(activities.energies.prices.price_ranges)
    _save_table(con, "price_range_percentiles", _dim_lookup_df([f"{p:.3f}" for p in price_range_pcts]).assign(pct=price_range_pcts), pk="idx")

    # === activities ===
    _save_table(con, "activities", _activities_dim_df(activity_entities), pk="name")

    _save_table(con, "activity_volumes",
        _long2(activity_names, "activity_name", periods, "period", activities.volumes, "volume"),
        pk=["activity_name", "period"], fks=[("activity_name", "activities", "name"), ("period", "periods", "period")])

    _save_table(con, "activity_prices_initialized",
        _long2(activity_names, "activity_name", periods, "period", activities.prices.initialized, "price"),
        pk=["activity_name", "period"], fks=[("activity_name", "activities", "name"), ("period", "periods", "period")])

    _save_table(con, "activity_prices_hourly",
        _long3(hours, "hour", activity_names, "activity_name", periods, "period", activities.prices.hourly, "price"),
        pk=["activity_name", "hour", "period"],
        fks=[("activity_name", "activities", "name"), ("period", "periods", "period")])

    _save_table(con, "activity_energy_scarcity",
        _long2(energy_names, "activity_name", periods, "period", activities.energies.scarcity, "scarcity"),
        pk=["activity_name", "period"], fks=[("activity_name", "activities", "name"), ("period", "periods", "period")])

    _save_table(con, "activity_energy_prices_yearly",
        _long2(energy_names, "activity_name", periods, "period", activities.energies.prices.yearly, "price"),
        pk=["activity_name", "period"], fks=[("activity_name", "activities", "name"), ("period", "periods", "period")])

    _save_table(con, "activity_energy_prices_hourly",
        _long3(hours, "hour", energy_names, "activity_name", periods, "period", activities.energies.prices.hourly, "price"),
        pk=["activity_name", "hour", "period"],
        fks=[("activity_name", "activities", "name"), ("period", "periods", "period")])

    _save_table(con, "activity_energy_price_ranges",
        _long3(range(dimensions['nRp']), "price_range_idx", energy_names, "activity_name", periods, "period",
               activities.energies.prices.ranges, "price"),
        pk=["price_range_idx", "activity_name", "period"],
        fks=[("price_range_idx", "price_range_percentiles", "idx"),
             ("activity_name", "activities", "name"), ("period", "periods", "period")])

    _save_table(con, "activity_emission_prices_yearly",
        _long2(emission_names, "activity_name", periods, "period", activities.emissions.prices.yearly, "price"),
        pk=["activity_name", "period"], fks=[("activity_name", "activities", "name"), ("period", "periods", "period")])

    # === technologies ===
    _save_table(con, "technologies", _technologies_dim_df(tech_entities), pk="id",
                fks=[("activity_name", "activities", "name")])

    _save_table(con, "technology_costs",
        pd.concat([
            _long2(tech_ids, "tech_id", periods, "period", technologies.balancers.costs.investments, "investment"),
            _long2(tech_ids, "tech_id", periods, "period", technologies.balancers.costs.foms, "fom")[["fom"]],
            _long2(tech_ids, "tech_id", periods, "period", technologies.balancers.costs.voms, "vom")[["vom"]],
        ], axis=1),
        pk=["tech_id", "period"], fks=[("tech_id", "technologies", "id"), ("period", "periods", "period")])

    _save_table(con, "technology_stock_limits",
        pd.concat([
            _long2(tech_ids, "tech_id", periods, "period", technologies.balancers.stocks.dec_planned, "dec_planned"),
            _long2(tech_ids, "tech_id", periods, "period", technologies.balancers.stocks.min, "min")[["min"]],
            _long2(tech_ids, "tech_id", periods, "period", technologies.balancers.stocks.max, "max")[["max"]],
        ], axis=1),
        pk=["tech_id", "period"], fks=[("tech_id", "technologies", "id"), ("period", "periods", "period")])

    _save_table(con, "technology_stock_evolution",
        _long2(tech_ids, "tech_id", periods, "period", technologies.balancers.stocks.evolution, "stock"),
        pk=["tech_id", "period"], fks=[("tech_id", "technologies", "id"), ("period", "periods", "period")])

    _save_table(con, "technology_investments",
        _long2(tech_ids, "tech_id", periods, "period", technologies.balancers.investments, "investment"),
        pk=["tech_id", "period"], fks=[("tech_id", "technologies", "id"), ("period", "periods", "period")])

    _save_table(con, "technology_decommissionings",
        _long2(tech_ids, "tech_id", periods, "period", technologies.balancers.decommissionings, "decommissioning"),
        pk=["tech_id", "period"], fks=[("tech_id", "technologies", "id"), ("period", "periods", "period")])

    _save_table(con, "technology_use",
        _long2(tech_ids, "tech_id", periods, "period", technologies.balancers.use.yearly, "use"),
        pk=["tech_id", "period"], fks=[("tech_id", "technologies", "id"), ("period", "periods", "period")])

    _save_table(con, "technology_use_hourly",
        _long3(hours, "hour", tech_ids, "tech_id", periods, "period", technologies.balancers.use.hourly, "use"),
        pk=["tech_id", "hour", "period"],
        fks=[("tech_id", "technologies", "id"), ("period", "periods", "period")])

    _save_table(con, "technology_lcops",
        _long2(tech_ids, "tech_id", periods, "period", technologies.balancers.lcops.values, "lcop_total"),
        pk=["tech_id", "period"], fks=[("tech_id", "technologies", "id"), ("period", "periods", "period")])

    _save_table(con, "technology_lcops_detail",
        _long3(tech_ids, "tech_id", range(dimensions['nTL']), "lcop_category_idx", periods, "period",
               technologies.balancers.lcops.matrix, "value"),
        pk=["tech_id", "lcop_category_idx", "period"],
        fks=[("tech_id", "technologies", "id"), ("lcop_category_idx", "lcop_categories", "idx"), ("period", "periods", "period")])

    _save_table(con, "technology_mca",
        _long3(tech_ids, "tech_id", range(dimensions['nMC']), "mca_category_idx", periods, "period",
               technologies.balancers.mca.matrix, "value"),
        pk=["tech_id", "mca_category_idx", "period"],
        fks=[("tech_id", "technologies", "id"), ("mca_category_idx", "mca_categories", "idx"), ("period", "periods", "period")])

    _save_table(con, "technology_choices_agent",
        _long3(tech_ids, "tech_id", range(dimensions['nAT']), "agent_type_idx", periods, "period",
               technologies.balancers.choices_agent, "value"),
        pk=["tech_id", "agent_type_idx", "period"],
        fks=[("tech_id", "technologies", "id"), ("agent_type_idx", "agent_types", "idx"), ("period", "periods", "period")])

    _save_table(con, "technology_choices_lcop_order",
        _long2(tech_ids, "tech_id", periods, "period", technologies.balancers.choices_lcop_order, "lcop_order"),
        pk=["tech_id", "period"], fks=[("tech_id", "technologies", "id"), ("period", "periods", "period")])

    _save_table(con, "technology_generators",
        pd.concat([
            _long2(tech_ids, "tech_id", periods, "period", technologies.balancers.generators.NUF, "nuf"),
            _long2(tech_ids, "tech_id", periods, "period", technologies.balancers.generators.CR, "capture_rate")[["capture_rate"]],
            _long2(tech_ids, "tech_id", periods, "period", technologies.balancers.generators.CF, "cash_flow")[["cash_flow"]],
        ], axis=1),
        pk=["tech_id", "period"], fks=[("tech_id", "technologies", "id"), ("period", "periods", "period")])

    _save_table(con, "technology_activity_balances",
        _long2(tech_ids, "tech_id", activity_names, "activity_name", technologies.balancers.activity_balances, "value"),
        pk=["tech_id", "activity_name"], fks=[("tech_id", "technologies", "id"), ("activity_name", "activities", "name")])

    # === infrastructure ===
    infra_df = pd.DataFrame({
        "id": [i.id for i in infra_entities],
        "seq": [i.idx for i in infra_entities],
        "category": [i.category for i in infra_entities],
        "name": [i.name for i in infra_entities],
        "unit": [i.unit for i in infra_entities],
        "activity_name": [i.activity_name for i in infra_entities],
        "cap2act": [i.cap2act for i in infra_entities],
        "lifetime": [i.lifetime for i in infra_entities],
        "stock_initial": [i.stock_initial for i in infra_entities],
    })
    _save_table(con, "infrastructure", infra_df, pk="id", fks=[("activity_name", "activities", "name")])

    _save_table(con, "infrastructure_costs",
        pd.concat([
            _long2(infra_ids, "infra_id", periods, "period", technologies.infra.costs.investments, "investment"),
            _long2(infra_ids, "infra_id", periods, "period", technologies.infra.costs.foms, "fom")[["fom"]],
        ], axis=1),
        pk=["infra_id", "period"], fks=[("infra_id", "infrastructure", "id"), ("period", "periods", "period")])

    _save_table(con, "infrastructure_stock_evolution",
        _long2(infra_ids, "infra_id", periods, "period", technologies.infra.stocks.evolution, "stock"),
        pk=["infra_id", "period"], fks=[("infra_id", "infrastructure", "id"), ("period", "periods", "period")])

    _save_table(con, "infrastructure_investments",
        _long2(infra_ids, "infra_id", periods, "period", technologies.infra.investments, "investment"),
        pk=["infra_id", "period"], fks=[("infra_id", "infrastructure", "id"), ("period", "periods", "period")])

    # === retrofittings ===
    retro = technologies.retrofittings
    _save_table(con, "retrofittings",
        pd.DataFrame({"from_tech": retro["from"], "to_tech": retro["to"], "cost": retro["costs"]}),
        pk=["from_tech", "to_tech"], fks=[("from_tech", "technologies", "id"), ("to_tech", "technologies", "id")])

    # === profiles ===
    _save_table(con, "profile_types", _dim_lookup_df(profiles.types), pk="idx")
    _save_table(con, "hourly_profiles",
        _long2(range(len(profiles.types)), "profile_type_idx", hours, "hour", profiles.shapes.T, "value"),
        pk=["profile_type_idx", "hour"], fks=[("profile_type_idx", "profile_types", "idx")])

    _save_table(con, "interconnectors", _dim_lookup_df(profiles.interconnectors), pk="idx")
    _save_table(con, "interconnector_prices",
        _long3(hours, "hour", range(len(profiles.interconnectors)), "interconnector_idx", periods, "period",
               profiles.prices, "price"),
        pk=["interconnector_idx", "hour", "period"],
        fks=[("interconnector_idx", "interconnectors", "idx"), ("period", "periods", "period")])

    # === results ===
    _save_table(con, "results_emissions",
        pd.DataFrame({"period": periods, "emissions": np.asarray(results.emissions).reshape(-1)}),
        pk="period", fks=[("period", "periods", "period")])

    _save_table(con, "results_emissions_sector",
        pd.concat([
            _long2(types.sectors, "sector", periods, "period", results.emissions_sector_pos, "positive"),
            _long2(types.sectors, "sector", periods, "period", results.emissions_sector_neg, "negative")[["negative"]],
        ], axis=1),
        pk=["sector", "period"], fks=[("sector", "type_sectors", "sector"), ("period", "periods", "period")])

    _save_table(con, "results_emissions_stored",
        pd.DataFrame({"period": periods, "stored": np.asarray(results.emissions_stored).reshape(-1)}),
        pk="period", fks=[("period", "periods", "period")])

    _save_table(con, "results_primary_energy",
        _long2(types.energy.labels, "energy_label", periods, "period", results.primary, "primary"),
        pk=["energy_label", "period"], fks=[("energy_label", "type_energy_labels", "label"), ("period", "periods", "period")])

    _save_table(con, "results_exports",
        _long2(energy_names, "activity_name", periods, "period", results.exports, "exports"),
        pk=["activity_name", "period"], fks=[("activity_name", "activities", "name"), ("period", "periods", "period")])

    _save_table(con, "results_costs",
        _long2(range(len(results.costs.categories)), "cost_category_idx", periods, "period", results.costs.system, "cost"),
        pk=["cost_category_idx", "period"], fks=[("cost_category_idx", "cost_categories", "idx"), ("period", "periods", "period")])

    _save_table(con, "results_policy_cashflows",
        _long2(types.policy_cashflows_categories, "cashflow_category", periods, "period", results.policy_cashflows, "cashflow"),
        pk=["cashflow_category", "period"],
        fks=[("cashflow_category", "type_policy_cashflow_categories", "category"), ("period", "periods", "period")])

    con.close()
    print(f"Full simulation state saved to {db_path}")


def save_excel_duckdb(types, activities, technologies, agents, results, db_path):
    """Persist exactly what results_write.py writes to the Excel reports,
    as relational tables (dimension tables for activities/technologies/
    periods + fact tables referencing them) instead of spreadsheet cells.
    """
    if os.path.exists(db_path):
        os.remove(db_path)
    con = duckdb.connect(db_path)

    periods = list(activities.periods)
    activity_entities = activities.entities
    activity_names = list(activities.names)
    tech_entities = technologies.balancers.entities
    tech_ids = [t.id for t in tech_entities]
    energy_activities = [a for a in activity_entities if a.is_energy]
    energy_names = [a.name for a in energy_activities]
    emission_activities = [a for a in activity_entities if a.is_emission]
    emission_names = [a.name for a in emission_activities]
    elec_activities = [a for a in activity_entities if a.is_electricity]
    gaseous_activities = [a for a in activity_entities if a.is_gaseous]

    # === dimension tables ===
    _save_table(con, "periods", _periods_df(periods), pk="period")
    _save_table(con, "activities", _activities_dim_df(activity_entities), pk="name")
    _save_table(con, "technologies", _technologies_dim_df(tech_entities), pk="id",
                fks=[("activity_name", "activities", "name")])
    _save_table(con, "mca_categories", _dim_lookup_df(agents.criteria.categories), pk="idx")
    _save_table(con, "agent_types", _dim_lookup_df(agents.types), pk="idx")
    _save_table(con, "lcop_categories", _dim_lookup_df(technologies.balancers.lcops.categories), pk="idx")

    # === system_costs (B€/y) ===
    _save_table(con, "system_costs",
        _long2(results.costs.categories, "cost_category", periods, "period", results.costs.system / 1000, "cost_beur"),
        pk=["cost_category", "period"], fks=[("period", "periods", "period")])

    # === system_emissions (Mton CO2eq) ===
    _save_table(con, "system_emissions",
        pd.DataFrame({"period": periods, "emissions_mton": np.asarray(results.emissions).reshape(-1)}),
        pk="period", fks=[("period", "periods", "period")])

    # === sectoral_emissions (Mton CO2eq) ===
    _save_table(con, "sectoral_emissions",
        pd.concat([
            _long2(types.sectors, "sector", periods, "period", results.emissions_sector_pos, "positive"),
            _long2(types.sectors, "sector", periods, "period", results.emissions_sector_neg, "negative")[["negative"]],
        ], axis=1),
        pk=["sector", "period"], fks=[("period", "periods", "period")])

    # === policy_cashflows (B€/y) ===
    _save_table(con, "policy_cashflows",
        _long2(types.policy_cashflows_categories, "cashflow_category", periods, "period",
               results.policy_cashflows / 1000, "cashflow_beur"),
        pk=["cashflow_category", "period"], fks=[("period", "periods", "period")])

    # === technology use/stock/investments ===
    _save_table(con, "technology_use",
        _long2(tech_ids, "tech_id", periods, "period", technologies.balancers.use.yearly, "use"),
        pk=["tech_id", "period"], fks=[("tech_id", "technologies", "id"), ("period", "periods", "period")])

    _save_table(con, "technology_stock",
        _long2(tech_ids, "tech_id", periods, "period", technologies.balancers.stocks.evolution, "stock"),
        pk=["tech_id", "period"], fks=[("tech_id", "technologies", "id"), ("period", "periods", "period")])

    _save_table(con, "technology_investments",
        _long2(tech_ids, "tech_id", periods, "period", technologies.balancers.investments, "investment"),
        pk=["tech_id", "period"], fks=[("tech_id", "technologies", "id"), ("period", "periods", "period")])

    # === sectoral balance (consumption/production per sector+energy label) ===
    coord_categories = np.array([t.category not in ('Primary', 'Emission', 'Exports') for t in tech_entities])
    tech_sector = np.array([t.sector for t in tech_entities])
    sectors = np.unique(tech_sector[coord_categories])
    energy_labels = types.energy.labels
    # Emission-type activities are now stored positive-for-emitters (IESA-Opt
    # convention). Most have energy_label != any real energy label so never
    # reach this table, but energy_label 'NA' is itself a real member of
    # types.energy.labels and all 10 emission activities carry it - restore
    # the old (negative-for-emitters) convention for this local copy first so
    # the consumption/production split below matches the old behavior.
    activity_balances = technologies.balancers.activity_balances.copy()
    coord_emission_act = np.array([a.is_emission for a in activity_entities])
    activity_balances[:, coord_emission_act] = -activity_balances[:, coord_emission_act]
    tech_use = technologies.balancers.use.yearly
    consumption_balance = activity_balances.copy()
    consumption_balance[activity_balances > 0] = 0
    production_balance = activity_balances.copy()
    production_balance[activity_balances < 0] = 0

    rows = []
    for s in sectors:
        for el in energy_labels:
            tech_coord = np.logical_and(coord_categories, tech_sector == s)
            act_coord = np.array([a.label == el for a in activity_entities])
            cons_sum = np.sum(consumption_balance[tech_coord][:, act_coord], axis=1)
            consumption = np.sum(tech_use[tech_coord, :] * cons_sum[:, None], axis=0)
            prod_sum = np.sum(production_balance[tech_coord][:, act_coord], axis=1)
            production = np.sum(tech_use[tech_coord, :] * prod_sum[:, None], axis=0)
            for iP, period in enumerate(periods):
                rows.append((s, el, period, consumption[iP], production[iP]))
    sector_balance_df = pd.DataFrame(rows, columns=["sector", "energy_label", "period", "consumption", "production"])
    _save_table(con, "sectoral_balance", sector_balance_df,
        pk=["sector", "energy_label", "period"], fks=[("period", "periods", "period")])

    # === energy / emission prices (yearly) ===
    _save_table(con, "energy_prices",
        _long2(energy_names, "activity_name", periods, "period", activities.energies.prices.yearly, "price"),
        pk=["activity_name", "period"], fks=[("activity_name", "activities", "name"), ("period", "periods", "period")])

    _save_table(con, "emission_prices",
        _long2(emission_names, "activity_name", periods, "period", activities.emissions.prices.yearly, "price"),
        pk=["activity_name", "period"], fks=[("activity_name", "activities", "name"), ("period", "periods", "period")])

    # === LCOPs / MCAs / agent choices ===
    _save_table(con, "technology_lcops",
        _long3(tech_ids, "tech_id", range(len(technologies.balancers.lcops.categories)), "lcop_category_idx", periods, "period",
               technologies.balancers.lcops.matrix, "value"),
        pk=["tech_id", "lcop_category_idx", "period"],
        fks=[("tech_id", "technologies", "id"), ("lcop_category_idx", "lcop_categories", "idx"), ("period", "periods", "period")])

    _save_table(con, "technology_mca",
        _long3(tech_ids, "tech_id", range(len(agents.criteria.categories)), "mca_category_idx", periods, "period",
               technologies.balancers.mca.matrix, "value"),
        pk=["tech_id", "mca_category_idx", "period"],
        fks=[("tech_id", "technologies", "id"), ("mca_category_idx", "mca_categories", "idx"), ("period", "periods", "period")])

    _save_table(con, "technology_agent_choices",
        _long3(tech_ids, "tech_id", range(len(agents.types)), "agent_type_idx", periods, "period",
               technologies.balancers.choices_agent, "value"),
        pk=["tech_id", "agent_type_idx", "period"],
        fks=[("tech_id", "technologies", "id"), ("agent_type_idx", "agent_types", "idx"), ("period", "periods", "period")])

    # === hourly power / gas prices (EUR/MWh) ===
    nH = activities.prices.hourly.shape[0]
    hours = list(range(nH))
    activities_elec_coord = activities.electricity.coords
    power_hourly = activities.prices.hourly[:, activities_elec_coord, :] * 3.6
    elec_names = [a.name for a in elec_activities]
    _save_table(con, "hourly_power_prices",
        _long3(hours, "hour", elec_names, "activity_name", periods, "period", power_hourly, "price_eur_mwh"),
        pk=["activity_name", "hour", "period"],
        fks=[("activity_name", "activities", "name"), ("period", "periods", "period")])

    activities_gaseous_coord = activities.gaseous.coords
    gaseous_hourly = activities.prices.hourly[:, activities_gaseous_coord, :]
    gaseous_names = [a.name for a in gaseous_activities]
    corr_units = np.array([1.0 if a.is_emission else 3.6 for a in gaseous_activities])
    gaseous_hourly = gaseous_hourly * corr_units[None, :, None]
    _save_table(con, "hourly_gas_prices",
        _long3(hours, "hour", gaseous_names, "activity_name", periods, "period", gaseous_hourly, "price_eur_mwh"),
        pk=["activity_name", "hour", "period"],
        fks=[("activity_name", "activities", "name"), ("period", "periods", "period")])

    con.close()
    print(f"Excel-equivalent results saved to {db_path}")
