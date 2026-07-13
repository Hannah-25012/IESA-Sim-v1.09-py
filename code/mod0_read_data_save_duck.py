# File to read data from the excel input file
from typing import *

import os
import pandas as pd
import numpy as np
import pickle
import duckdb
from Constants import Parameters
import re
# FIX: To suppress warning "UserWarning: Data Validation extension is not supported and will be removed warn(msg)" - not sure what to do with this, maybe fix later
import warnings

from Constants.Parameters import Activities, Agents, Types, Technologies, Infrastructure, Retrofitting, HourlyProfiles, PriceProfiles

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


def dict_to_df_padded_zero(data):
    max_len = max(len(v) for v in data.values())
    padded = {
        k: list(v) + [0] * (max_len - len(v))
        for k, v in data.items()
    }
    return pd.DataFrame(padded)

def rename_volumes_col(col):
    if str(col).startswith(Parameters.Activities.periods_start):
        year_match = re.search(r'\d+', str(col))
        if year_match:
            return f"volumes_{year_match.group()}"
    return col  # leave unchanged if it doesn't match

def flatten_header(columns):
    cols = [
        " / ".join(str(x) for x in col if pd.notna(x) and not str(x).startswith("Unnamed"))
        for col in columns
    ]
    return [re.sub(r'\s*/?\s*\[.*\]\s*$', '', col).strip() for col in cols]

def dtype_to_sql(dtype):
    if "bool" in str(dtype):
        return "BOOLEAN"
    elif "int" in str(dtype):
        return "INTEGER"
    elif "float" in str(dtype):
        return "DOUBLE"
    else:
        return "VARCHAR"

def build_create_table_sql(table_name, df, pk=None, fks=None):
    cols_sql = ",\n    ".join(f'"{col}" {dtype_to_sql(dt)}' for col, dt in df.dtypes.items())
    constraints = []
    if pk:
        pk_cols = pk if isinstance(pk, str) else ", ".join(pk)
        constraints.append(f"PRIMARY KEY ({pk_cols})")
    for col, ref_table, ref_col in (fks or []):
        constraints.append(f'FOREIGN KEY ("{col}") REFERENCES {ref_table}({ref_col})')
    parts = [cols_sql] + constraints
    return f"CREATE TABLE {table_name} (\n    " + ",\n    ".join(parts) + "\n)"

def matrix_to_long_df(row_ids, row_col_name, matrix, col_ids, col_col_name, value_col_name):
    idx = pd.MultiIndex.from_product([row_ids, col_ids], names=[row_col_name, col_col_name])
    return pd.DataFrame({value_col_name: np.asarray(matrix).reshape(-1)}, index=idx).reset_index()

def build_policy_long_df(policy_matrix, periods):
    activity_names = policy_matrix[:, 0]
    units = policy_matrix[:, 1]
    values = policy_matrix[:, 2:2 + len(periods)].astype(float)
    idx = pd.MultiIndex.from_product([range(len(activity_names)), periods], names=["row", "period"])
    long_df = pd.DataFrame({"value": values.reshape(-1)}, index=idx).reset_index()
    long_df["activity_name"] = long_df["row"].map(lambda i: activity_names[i])
    long_df["unit"] = long_df["row"].map(lambda i: units[i])
    return long_df[["activity_name", "unit", "period", "value"]]

def mod0_read_data_save_duck(file_name):

    # Create empty dictionaries to store data that is derived from excel sheets
    parameters, types, activities, profiles, technologies, agents, policies = {}, {}, {}, {}, {}, {}, {}

    db_path = "SIMmodel.duckdb"
    if os.path.exists(db_path):
        os.remove(db_path)
    con = duckdb.connect(db_path)
    # === Parameters sheet ===
    print('--Reading parameters sheet')
    parameters_values = pd.read_excel(file_name, sheet_name='Parameters', usecols="B", skiprows=2, nrows=20).squeeze()

    parameters_input = pd.read_excel(
        file_name,
        sheet_name='Parameters',
        skiprows=2
    )

    # Extract individual parameters

    powinv_SPBT_benchmark = parameters_input.loc[parameters_input["Name"] == Parameters.Parameters.powinv_SPBT_benchmark, "Value"].values[0]
    powinv_SPBT_min =parameters_input.loc[parameters_input["Name"] == Parameters.Parameters.powinv_SPBT_min, "Value"].values[0]
    powinv_CR_threshold = parameters_input.loc[parameters_input["Name"] == Parameters.Parameters.powinv_CR_threshold, "Value"].values[0]
    powinv_CR_min = parameters_input.loc[parameters_input["Name"] == Parameters.Parameters.powinv_CR_min, "Value"].values[0]
    powinv_NUF_threshold = parameters_input.loc[parameters_input["Name"] == Parameters.Parameters.powinv_NUF_threshold, "Value"].values[0]
    powinv_NUF_min = parameters_input.loc[parameters_input["Name"] == Parameters.Parameters.powinv_NUF_min, "Value"].values[0]
    scarcity_penalization = parameters_input.loc[parameters_input["Name"] == Parameters.Parameters.scarcity_penalization, "Value"].values[0]
    gas_premium = parameters_input.loc[parameters_input["Name"] == Parameters.Parameters.gas_premium, "Value"].values[0]
    voll_value = parameters_input.loc[parameters_input["Name"] == Parameters.Parameters.voll_value, "Value"].values[0]
    voll_factor = Parameters.Parameters.voll_factor
    voll = voll_value / voll_factor
    gov_dr = parameters_input.loc[parameters_input["Name"] == Parameters.Parameters.gov_dr, "Value"].values[0]
    exports_value = parameters_input.loc[parameters_input["Name"] == Parameters.Parameters.exports_value, "Value"].values[0]
    min_spread_value = parameters_input.loc[parameters_input["Name"] == Parameters.Parameters.min_spread_value, "Value"].values[0]
    min_spread_factor = Parameters.Parameters.min_spread_factor
    min_spread = min_spread_value / min_spread_factor

    # === Parameters (in-memory dict, matches mod0_read_data.py's structure) ===
    parameters['powinv'] = {
        'SPBT_benchmark': powinv_SPBT_benchmark,
        'SPBT_min': powinv_SPBT_min,
        'CR_threshold': powinv_CR_threshold,
        'CR_min': powinv_CR_min,
        'NUF_threshold': powinv_NUF_threshold,
        'NUF_min': powinv_NUF_min,
    }
    parameters['scarcity'] = {
        'penalization': scarcity_penalization,
        'gas_premium': gas_premium
    }
    parameters['voll'] = voll
    parameters['min_spread'] = min_spread
    parameters['gov_dr'] = gov_dr
    parameters['exports_value'] = exports_value

    parameters_powinv = pd.DataFrame(columns=["Name", "Value"])
    # Store parameters in

    rows = []
    rows.append({"Name": "SPBT_benchmark", "Value": powinv_SPBT_benchmark})
    rows.append({"Name": "SPBT_min", "Value": powinv_SPBT_min})
    rows.append({"Name": "CR_threshold", "Value": powinv_CR_threshold})
    rows.append({"Name": "CR_min", "Value": powinv_CR_min})
    rows.append({"Name": "NUF_threshold", "Value": powinv_NUF_threshold})
    rows.append({"Name": "NUF_min", "Value": powinv_NUF_min})
    parameters_powinv = pd.DataFrame(rows, columns=["Name", "Value"])

    parameters_scarcity = pd.DataFrame(columns=["Name", "Value"])
    # Store parameters in

    rows = []
    rows.append({"Name": "penalization", "Value": scarcity_penalization})
    rows.append({"Name": "gas_premium", "Value": gas_premium})
    parameters_scarcity = pd.DataFrame(rows, columns=["Name", "Value"])

    rows = []
    rows.append({"Name": "voll", "Value": voll})
    rows.append({"Name": "min_spread", "Value": min_spread})
    rows.append({"Name": "gov_dr", "Value": gov_dr})
    rows.append({"Name": "exports_value", "Value": exports_value})
    parameters_input_short = pd.DataFrame(rows, columns=["Name", "Value"])

    df_parameters_parameters_scarcity = pd.DataFrame(parameters_scarcity)
    try:
        con.execute("CREATE TABLE original_params AS SELECT * FROM parameters_input")
        con.execute("ALTER TABLE original_params ADD PRIMARY KEY (Name)")
        con.execute("CREATE TABLE original_params_short AS SELECT * FROM parameters_input_short")
        con.execute("ALTER TABLE original_params_short ADD PRIMARY KEY (Name)")
        con.execute("CREATE TABLE powinv AS SELECT * FROM parameters_powinv")
        con.execute("ALTER TABLE powinv ADD PRIMARY KEY (Name)")
        con.execute("CREATE TABLE scarcity AS SELECT * FROM parameters_scarcity")
        con.execute("ALTER TABLE scarcity ADD PRIMARY KEY (Name)")
    except:
        print("error in saving Parameters to duckdb")

    # === Types sheet ===
    print('--Reading types sheet')
    # Row below the header (excel row 3) is a unit-annotation row, not data, so column
    # positions are resolved from the header names and then read with the original
    # skiprows/nrows that skip past it.
    types_header = pd.read_excel(file_name, sheet_name='Types', header=1, nrows=0).columns
    activity_type = pd.read_excel(file_name, sheet_name='Types', usecols=[types_header.get_loc(Types.activity_type)], skiprows=2, nrows=4).squeeze().tolist()
    sectors = pd.read_excel(file_name, sheet_name='Types', usecols=[types_header.get_loc(Types.sectors)], skiprows=2, nrows=27).dropna().squeeze().tolist()
    energy_labels = pd.read_excel(file_name, sheet_name='Types', usecols=[types_header.get_loc(Types.energy_labels)], skiprows=2, nrows=17, dtype=str, keep_default_na=False).squeeze().tolist()
    energy_price_init = pd.read_excel(file_name, sheet_name='Types', usecols=[types_header.get_loc(Types.energy_price_init)], skiprows=2, nrows=27).dropna().squeeze().to_numpy()

    # Store in dictionary
    types['activities'] = activity_type
    types['sectors'] = sectors
    types['energy'] = {
        'labels' : energy_labels,
        'price init' : energy_price_init,
    }
    df_activities = pd.DataFrame(activity_type, columns=["activity_type"])
    df_sectors = pd.DataFrame(activity_type, columns=["sectors"])
    df_energy = dict_to_df_padded_zero(types['energy'])
    try:
        con.execute("CREATE TABLE activity_types AS SELECT * FROM df_activities")
        con.execute("ALTER TABLE activity_types ADD PRIMARY KEY (activity_type)")
        con.execute("CREATE TABLE sectors AS SELECT * FROM df_sectors")
        con.execute("ALTER TABLE sectors ADD PRIMARY KEY (sectors)")
        con.execute("CREATE TABLE energy_types AS SELECT * FROM df_energy")
        con.execute("ALTER TABLE energy_types ADD PRIMARY KEY (labels)")
    except:
        print("error in saving types to duckdb")

    # === Agents sheet ===
    #Agents need to be before activities because they are foreign keys
    print('--Reading agents sheet')

    agents = pd.read_excel(
        file_name,
        sheet_name='Agents',
        skiprows=1,
        header=[0, 1,2],
    )
    agents.columns = flatten_header(agents.columns)
    agents = agents.rename(columns={Agents.rates: "rates"})
    agents = agents.rename(columns={Agents.profiles: "Name"})
    # 1. select columns that start with the prefix
    selected_cols = [col for col in agents.columns if str(col).startswith(Agents.types)]
    # 2. extract the agent types
    agent_type_names = [str(col)[len(Agents.types):].strip(" /").strip() for col in selected_cols]
    types_df = pd.DataFrame(agent_type_names, columns=["Name"])

    rename_map = dict(zip(selected_cols, agent_type_names))
    agents = agents.rename(columns=rename_map)

    agent_profiles = agents[agents["rates"].notna()]
    agent_profiles["rates"] = agent_profiles["rates"] / 100
    agent_profiles_toTable = agent_profiles[["Name", "rates"]]
    population_df = agent_profiles.melt(
        id_vars=["Name"],  # FK to agent_profiles
        value_vars=agent_type_names,  # the agent-type columns: Innovators, Early adopters, Majority, Laggards
        var_name="agent_type",  # FK to agent_types
        value_name="value"
    ).rename(columns={"Name": "agent_profile"})
    population_df["value"]=population_df["value"]/100

    try:
        con.execute("CREATE TABLE agent_profiles AS SELECT * FROM agent_profiles_toTable")
        con.execute("ALTER TABLE agent_profiles ADD PRIMARY KEY (Name)")
        con.execute("CREATE TABLE agent_types AS SELECT * FROM types_df")
        con.execute("ALTER TABLE agent_types ADD PRIMARY KEY (Name)")

    except:
        print("error in saving agent_profiles and types to duckdb")

    try:
        con.execute("""
            CREATE TABLE population (
                agent_profile VARCHAR,
                agent_type VARCHAR,
                value DOUBLE,
                FOREIGN KEY (agent_profile) REFERENCES agent_profiles(Name),
                FOREIGN KEY (agent_type) REFERENCES agent_types(Name)
            )
        """)
        con.execute("INSERT INTO population SELECT * FROM population_df")
    except:
        print("error in saving populations to duckdb")

    weight_profiles = agents[agents["rates"].isna()]
    mask = weight_profiles.astype(str).apply(lambda row: row.str.contains(r"\[ad\]", regex=True, na=False)).any(axis=1)
    weight_profiles = weight_profiles[~mask]
    weight_profiles = weight_profiles.drop(columns=["rates"])
    wp=weight_profiles[['Name']]

    try:
        con.execute("CREATE TABLE agent_criteria AS SELECT * FROM wp")
        con.execute("ALTER TABLE agent_criteria ADD PRIMARY KEY (Name)")
    except:
        print("error in saving agent_profiles and types to duckdb")

    criteria_weights_df = weight_profiles.melt(
        id_vars=["Name"],  # FK to agent_criteria
        value_vars=agent_type_names,  # the agent-type columns: Innovators, Early adopters, Majority, Laggards
        var_name="agent_type",  # FK to agent_types
        value_name="value"
    ).rename(columns={"Name": "agent_criteria"})

    try:
        con.execute("""
            CREATE TABLE criteria_weights (
                agent_criteria VARCHAR,
                agent_type VARCHAR,
                value DOUBLE,
                FOREIGN KEY (agent_criteria) REFERENCES agent_criteria(Name),
                FOREIGN KEY (agent_type) REFERENCES agent_types(Name)
            )
        """)
        con.execute("INSERT INTO criteria_weights SELECT * FROM criteria_weights_df")
    except:
        print("error in saving criteria_weights_df to duckdb")

    # Reuse the already name-based parse above instead of re-reading the sheet by column letter
    agent_types = agent_type_names
    multiCriteria_categories = wp["Name"].tolist()
    agents_dr = agent_profiles_toTable["rates"].to_numpy()
    agents_populations = agent_profiles[agent_type_names].fillna(0).to_numpy() / 100
    weights_multiCriteria = weight_profiles[agent_type_names].fillna(0).to_numpy()
    agent_profiles = agent_profiles_toTable["Name"].tolist()

    # Store in dictionary
    agents = {
        'types': agent_types,
        'profiles': agent_profiles,
        'criteria': {
            'categories': multiCriteria_categories,
            'weights': weights_multiCriteria
        },
        'populations': agents_populations,
        'rates': agents_dr
    }

    # === Activities sheet ===
    print('--Reading activities sheet')
    activities_df = pd.read_excel(
        file_name,
        sheet_name='Activities',
        skiprows=6,
        header=[0, 1],
        keep_default_na=False,
    )
    activities_df.columns = flatten_header(activities_df.columns)

    prefix = Parameters.Activities.periods_start

    # 1. select columns that start with the prefix
    selected_cols = [col for col in activities_df.columns if str(col).startswith(prefix)]

    # 2. extract the years from those column names, and cast to int
    periods = [int(re.search(r'\d+', str(col)[len(prefix):]).group()) for col in selected_cols]

    print(selected_cols)
    print(periods)
    volume_cols = [rename_volumes_col(col) for col in selected_cols]
    activities_df.columns = [rename_volumes_col(col) for col in activities_df.columns]

    activities_df = activities_df.rename(columns={Activities.activities_names: "Name"})
    activities_df = activities_df.rename(columns={Activities.activity_resolution: "activity_resolution"})
    activities_df = activities_df.rename(columns={Activities.activity_type_act: "activity_type"})
    activities_df = activities_df.rename(columns={Activities.activity_label: "energy_label"})
    activities_df = activities_df.rename(columns={Activities.activity_agent: "agent_profile"})

    try:
        cols_sql = ",\n    ".join(f'"{col}" {dtype_to_sql(dt)}' for col, dt in activities_df.dtypes.items())

        create_stmt = f"""
        CREATE TABLE activities (
            {cols_sql},
            FOREIGN KEY (activity_type) REFERENCES activity_types(activity_type),
            FOREIGN KEY (agent_profile) REFERENCES agent_profiles(Name),
            FOREIGN KEY (energy_label) REFERENCES energy_types(labels)
        )
        """
        con.execute(create_stmt)
        con.execute("INSERT INTO activities SELECT * FROM activities_df")
        con.execute("ALTER TABLE activities ADD PRIMARY KEY (Name)")
    except Exception as e:
        print(f"error in saving activities to duckdb: {e}")

    # === Periods (derived) ===
    print('--Saving periods to duckdb')
    try:
        periods_df = pd.DataFrame({"period": periods, "period_order": list(range(len(periods)))})
        con.execute(build_create_table_sql("periods", periods_df, pk="period"))
        con.execute("INSERT INTO periods SELECT * FROM periods_df")
    except Exception as e:
        print(f"error in saving periods to duckdb: {e}")

    # === Activities (in-memory dict, matches mod0_read_data.py's structure) ===
    activity_volumes = activities_df[volume_cols].fillna(0).to_numpy()
    activities = {
        'names': activities_df["Name"].tolist(),
        'periods': np.array(periods),
        'volumes': activity_volumes,
        'resolution': activities_df["activity_resolution"].tolist(),
        'types': activities_df["activity_type"].tolist(),
        'labels': activities_df["energy_label"].tolist(),
        'agents': activities_df["agent_profile"].tolist(),
        'drivers': {},
        'energies': {},
        'emissions': {},
        'electricity': {},
        'gaseous': {},
        'infra': {},
        'prices': {},
    }

    # === Hourly profiles sheet ===
    print('--Reading hourly profiles sheet')
    hourly_header = pd.read_excel(file_name, sheet_name='HourlyProfiles', header=2, nrows=0).columns.tolist()
    hourly_profile_start = hourly_header.index(HourlyProfiles.month) + 1
    profile_types = hourly_header[hourly_profile_start:]
    # header=2 only consumes the profile-name row; the sheet also has a "Data source"
    # annotation row below it before the data starts, so skiprows must skip past that too.
    hourly_profiles = pd.read_excel(
        file_name, sheet_name='HourlyProfiles', header=None, skiprows=4, nrows=8760,
        usecols=range(hourly_profile_start, len(hourly_header))
    ).to_numpy()

    # Store in dictionary
    profiles['types'] = profile_types
    profiles['shapes'] = hourly_profiles

    print('--Saving hourly profiles to duckdb')
    try:
        profile_types_df = pd.DataFrame({"name": profile_types})
        con.execute(build_create_table_sql("hourly_profile_types", profile_types_df, pk="name"))
        con.execute("INSERT INTO hourly_profile_types SELECT * FROM profile_types_df")

        hourly_profiles_long = matrix_to_long_df(
            range(hourly_profiles.shape[0]), "hour", hourly_profiles, profile_types, "profile_type", "value"
        )
        con.execute(build_create_table_sql(
            "hourly_profiles", hourly_profiles_long,
            pk=["hour", "profile_type"],
            fks=[("profile_type", "hourly_profile_types", "name")]
        ))
        con.execute("INSERT INTO hourly_profiles SELECT * FROM hourly_profiles_long")
    except Exception as e:
        print(f"error in saving hourly profiles to duckdb: {e}")

    # === Price profiles sheet ===
    print('--Reading price profiles sheet')
    price_header = flatten_header(
        pd.read_excel(file_name, sheet_name='PriceProfiles', header=[1, 2], nrows=0).columns
    )
    leading_cols = {HourlyProfiles.hour, HourlyProfiles.day, HourlyProfiles.month}
    # Each interconnector contributes one "<name> / <year>" column per period; discover the
    # distinct interconnector names (in sheet order) instead of assuming a fixed "D:J" width.
    interconnector = []
    for name in price_header:
        if name in leading_cols:
            continue
        ic_name = name.rsplit(" / ", 1)[0]
        if ic_name not in interconnector:
            interconnector.append(ic_name)
    nIC = len(interconnector)

    price_positions = [price_header.index(f"{ic} / {p}") for ic in interconnector for p in periods]
    # Same "Data source" annotation row as HourlyProfiles sits below the header here too.
    price_profiles_raw = pd.read_excel(
        file_name, sheet_name='PriceProfiles', header=None, skiprows=4, nrows=8760, usecols=price_positions
    ).to_numpy()

    price_profiles = np.zeros((8760, nIC, len(periods)))
    for i in range(nIC):
        price_profiles[:, i, :] = price_profiles_raw[:, len(periods) * i:len(periods) * (i + 1)]

    # Store in dictionary
    profiles['interconnectors'] = interconnector
    profiles['prices'] = price_profiles

    print('--Saving price profiles to duckdb')
    try:
        interconnectors_df = pd.DataFrame({"id": range(nIC), "name": interconnector})
        con.execute(build_create_table_sql("interconnectors", interconnectors_df, pk="id"))
        con.execute("INSERT INTO interconnectors SELECT * FROM interconnectors_df")

        price_idx = pd.MultiIndex.from_product(
            [range(price_profiles.shape[0]), range(nIC), periods],
            names=["hour", "interconnector_id", "period"]
        )
        price_profiles_long = pd.DataFrame({"price": price_profiles.reshape(-1)}, index=price_idx).reset_index()
        con.execute(build_create_table_sql(
            "price_profiles", price_profiles_long,
            pk=["hour", "interconnector_id", "period"],
            fks=[("interconnector_id", "interconnectors", "id"), ("period", "periods", "period")]
        ))
        con.execute("INSERT INTO price_profiles SELECT * FROM price_profiles_long")
    except Exception as e:
        print(f"error in saving price profiles to duckdb: {e}")

    # === Technologies sheet ===
    print('--Reading technologies sheet')
    # Row above the data (excel row 5) is a mandatory/optional annotation row, not part of
    # the header, so column positions are resolved from the flattened group/field header
    # and then read with the original skiprows/nrows that skip past it.
    tech_index = pd.Index(flatten_header(
        pd.read_excel(file_name, sheet_name='Technologies', header=[1, 2, 3], nrows=0).columns
    ))

    def tech_col(name):
        return tech_index.get_loc(name)

    def tech_year_cols(base_name):
        return [tech_index.get_loc(f"{base_name} / {p}") for p in periods]

    def read_tech(usecols, nrows=793, **kwargs):
        return pd.read_excel(file_name, sheet_name='Technologies', usecols=usecols, skiprows=5, nrows=nrows, **kwargs)

    tech_balancers = read_tech([tech_col(Technologies.tech_id)]).dropna().squeeze().tolist()
    tech_names = read_tech([tech_col(Technologies.name)]).dropna().squeeze().tolist()
    tech_sector = read_tech([tech_col(Technologies.sector)]).dropna().squeeze().tolist()
    tech_subsector = read_tech([tech_col(Technologies.subsector)]).dropna().squeeze().tolist()
    tech_units = read_tech([tech_col(Technologies.unit)]).dropna().squeeze().tolist()
    activity_per_tech = read_tech([tech_col(Technologies.main_activity)]).dropna().squeeze().tolist()
    tech_categories = read_tech([tech_col(Technologies.category)]).dropna().squeeze().tolist()

    inv_cost = read_tech(tech_year_cols(Technologies.investment)).fillna(0).to_numpy()
    fom_cost = read_tech(tech_year_cols(Technologies.fixed_om)).fillna(0).to_numpy()
    vom_cost = read_tech(tech_year_cols(Technologies.variable_om)).fillna(0).to_numpy()

    ec_lifetime = read_tech([tech_col(Technologies.ec_lifetime)]).fillna(0).squeeze().to_numpy()
    cap2act = read_tech([tech_col(Technologies.cap2act)]).fillna(0).squeeze().to_numpy()

    dispatch_type_tech = read_tech([tech_col(Technologies.dispatch_type)]).dropna().squeeze().tolist()
    hourly_profile_tech = read_tech([tech_col(Technologies.hourly_profile)]).dropna().squeeze().tolist()
    social_perception_tech = read_tech([tech_col(Technologies.social_perception)]).dropna().squeeze().tolist()
    perceived_complexity_tech = read_tech([tech_col(Technologies.perceived_complexity)]).dropna().squeeze().tolist()

    subsidy_subject = read_tech([tech_col(Technologies.subsidy_subject)]).fillna(0).squeeze().to_numpy().astype(bool)
    feedin_subject = read_tech([tech_col(Technologies.feedin_subject)]).fillna(0).squeeze().to_numpy().astype(bool)

    shedding_capacity = read_tech([tech_col(Technologies.shedding_capacity)]).fillna(0).squeeze().to_numpy()
    shedding_limits = read_tech([tech_col(Technologies.shedding_volume)]).fillna(0).squeeze().to_numpy()
    shedding_guarantee = read_tech([tech_col(Technologies.shedding_guarantee)]).fillna(0).squeeze().to_numpy()

    flexibility_form = read_tech([tech_col(Technologies.flexibility_form)]).dropna().squeeze().tolist()
    flexibility_activity = read_tech([tech_col(Technologies.flexibility_activity)]).dropna().squeeze().tolist() # CHECK: flexibility_activity only retrieves 38 values and skips all empty rows. I don't know if that's a problem - compare with matlab
    flexibility_capacity = read_tech([tech_col(Technologies.flexibility_capacity)]).fillna(0).squeeze().to_numpy()
    flexibility_volume = read_tech([tech_col(Technologies.flexibility_volume)]).fillna(0).squeeze().to_numpy()
    flexibility_range = read_tech([tech_col(Technologies.flexibility_range)]).fillna(0).squeeze().tolist()
    flexibility_losses = read_tech([tech_col(Technologies.flexibility_losses)]).fillna(0).squeeze().to_numpy()
    flexibility_nonnegotiable = read_tech([tech_col(Technologies.flexibility_nonnegotiable)]).fillna(0).squeeze().to_numpy()

    buffer_up = read_tech([tech_col(Technologies.buffer_up)]).fillna(0).squeeze().to_numpy()
    buffer_down = read_tech([tech_col(Technologies.buffer_down)]).fillna(0).squeeze().to_numpy()
    buffer_capacity = read_tech([tech_col(Technologies.buffer_capacity)]).fillna(0).squeeze().to_numpy()

    tech_stock_deploy = read_tech([tech_col(Technologies.tech_stock_deploy)]).fillna(0).squeeze().to_numpy()
    tech_stock_exist = read_tech([tech_col(Technologies.tech_stock_exist)]).fillna(0).squeeze().to_numpy()
    # Derive tech_stock_dec in correct format
    nTb = len(tech_balancers)
    nP = len(periods)
    # The planned-decommissioning / min-stock / max-stock blocks that follow have no
    # distinct column name of their own in the sheet (only ordinal or data-like
    # sub-headers), so their positions are derived from the last named anchor column
    # instead of a hardcoded Excel letter range.
    stock_block_start = tech_col(Technologies.tech_stock_exist) + 1
    dec_cols = list(range(stock_block_start, stock_block_start + (nP - 1)))
    min_cols = list(range(dec_cols[-1] + 1, dec_cols[-1] + 1 + nP))
    max_cols = list(range(min_cols[-1] + 1, min_cols[-1] + 1 + nP))
    raw_data = read_tech(dec_cols).fillna(0).to_numpy()
    tech_stock_dec = np.zeros((nTb, nP))
    tech_stock_dec[:, 1:] = raw_data
    tech_stock_min = read_tech(min_cols).fillna(0).to_numpy()
    tech_stock_max = read_tech(max_cols).fillna(0).to_numpy()

    # Store in dictionary
    technologies['balancers'] = {
        'ids': tech_balancers,
        'names': tech_names,
        'sectors': tech_sector,
        'subsectors': tech_subsector,
        'units': tech_units,
        'activities': activity_per_tech,
        'categories': tech_categories,
        'costs': {
            'investments': inv_cost,
            'foms': fom_cost,
            'voms': vom_cost,
            'lifetimes': ec_lifetime
        },
        'cap2acts': cap2act,
        'dispatch': dispatch_type_tech,
        'profiles': hourly_profile_tech,
        'agents': {
            'social_perception': social_perception_tech,
            'complexity': perceived_complexity_tech
        },
        'policies': {
            'subsidy_subject': subsidy_subject,
            'feedin_subject': feedin_subject
        },
        'shedding': {
            'capacity': shedding_capacity,
            'limits': shedding_limits,
            'guarantee': shedding_guarantee
        },
        'flexibility': {
            'form': flexibility_form,
            'activity': flexibility_activity,
            'capacity': flexibility_capacity,
            'volume': flexibility_volume,
            'range': flexibility_range,
            'losses': flexibility_losses,
            'nonnegotiable': flexibility_nonnegotiable
        },
        'buffers': {
            'up': buffer_up,
            'down': buffer_down,
            'capacity': buffer_capacity
        },
        'stocks': {
            'deploy': tech_stock_deploy,
            'initial': tech_stock_exist,
            'dec_planned': tech_stock_dec,
            'min': tech_stock_min,
            'max': tech_stock_max
        },
        'drivers' : {},
        'energies' : {},
        'emissions' : {},
        'use' : {},
        'investments' : {},
        'retrofittings' : {},
        'decommissionings' : {},
        'generators' : {},
        'loops' : {},
        'mca' : {},
    }

    print('--Saving technologies to duckdb')
    try:
        technologies_df = pd.DataFrame({
            "id": tech_balancers,
            "category": tech_categories,
            "sector": tech_sector,
            "subsector": tech_subsector,
            "name": tech_names,
            "unit": tech_units,
            "activity": activity_per_tech,
            "cap2act": cap2act,
            "lifetime": ec_lifetime,
            "dispatch_type": dispatch_type_tech,
            "hourly_profile": hourly_profile_tech,
            "social_perception": social_perception_tech,
            "perceived_complexity": perceived_complexity_tech,
            "subsidy_subject": subsidy_subject,
            "feedin_subject": feedin_subject,
            "shedding_capacity": shedding_capacity,
            "shedding_limits": shedding_limits,
            "shedding_guarantee": shedding_guarantee,
            "flexibility_form": flexibility_form,
            "flexibility_capacity": flexibility_capacity,
            "flexibility_volume": flexibility_volume,
            "flexibility_range": flexibility_range,
            "flexibility_losses": flexibility_losses,
            "flexibility_nonnegotiable": flexibility_nonnegotiable,
            "buffer_up": buffer_up,
            "buffer_down": buffer_down,
            "buffer_capacity": buffer_capacity,
            "stock_deploy": tech_stock_deploy,
            "stock_initial": tech_stock_exist,
        })
        con.execute(build_create_table_sql(
            "technologies", technologies_df, pk="id",
            fks=[("activity", "activities", "Name"), ("hourly_profile", "hourly_profile_types", "name")]
        ))
        con.execute("INSERT INTO technologies SELECT * FROM technologies_df")

        # Sparse: only technologies for which a flexible-activity coupling is defined (see CHECK note above)
        flexibility_activity_full = read_tech([tech_col(Technologies.flexibility_activity)], nrows=nTb).squeeze()
        flex_rows = [
            {"tech_id": tech_balancers[i], "activity_name": flexibility_activity_full.iloc[i]}
            for i in range(nTb) if pd.notna(flexibility_activity_full.iloc[i])
        ]
        technology_flexibility_activities_df = pd.DataFrame(flex_rows, columns=["tech_id", "activity_name"])
        con.execute(build_create_table_sql(
            "technology_flexibility_activities", technology_flexibility_activities_df, pk="tech_id",
            fks=[("tech_id", "technologies", "id"), ("activity_name", "activities", "Name")]
        ))
        con.execute("INSERT INTO technology_flexibility_activities SELECT * FROM technology_flexibility_activities_df")

        tech_costs_idx = pd.MultiIndex.from_product([tech_balancers, periods], names=["tech_id", "period"])
        technology_costs_df = pd.DataFrame({
            "investment": inv_cost.reshape(-1),
            "fom": fom_cost.reshape(-1),
            "vom": vom_cost.reshape(-1),
        }, index=tech_costs_idx).reset_index()
        con.execute(build_create_table_sql(
            "technology_costs", technology_costs_df, pk=["tech_id", "period"],
            fks=[("tech_id", "technologies", "id"), ("period", "periods", "period")]
        ))
        con.execute("INSERT INTO technology_costs SELECT * FROM technology_costs_df")

        tech_stocks_idx = pd.MultiIndex.from_product([tech_balancers, periods], names=["tech_id", "period"])
        technology_stocks_df = pd.DataFrame({
            "dec_planned": tech_stock_dec.reshape(-1),
            "min": tech_stock_min.reshape(-1),
            "max": tech_stock_max.reshape(-1),
        }, index=tech_stocks_idx).reset_index()
        con.execute(build_create_table_sql(
            "technology_stocks", technology_stocks_df, pk=["tech_id", "period"],
            fks=[("tech_id", "technologies", "id"), ("period", "periods", "period")]
        ))
        con.execute("INSERT INTO technology_stocks SELECT * FROM technology_stocks_df")
    except Exception as e:
        print(f"error in saving technologies to duckdb: {e}")

    # === Infrastructure sheet ===
    print('--Reading infrastructure sheet')
    infra_index = pd.Index(flatten_header(
        pd.read_excel(file_name, sheet_name='Infrastructure', header=[1, 2, 3], nrows=0).columns
    ))

    def infra_col(name):
        return infra_index.get_loc(name)

    def infra_year_cols(base_name):
        return [infra_index.get_loc(f"{base_name} / {p}") for p in periods]

    def read_infra(usecols, nrows=15, **kwargs):
        return pd.read_excel(file_name, sheet_name='Infrastructure', usecols=usecols, skiprows=4, nrows=nrows, **kwargs)

    tech_infra = read_infra([infra_col(Infrastructure.tech_id)]).dropna().squeeze().tolist()
    tech_categories_infra = read_infra([infra_col(Infrastructure.category)]).dropna().squeeze().tolist()
    tech_names_infra = read_infra([infra_col(Infrastructure.name)]).dropna().squeeze().tolist()
    tech_units_infra = read_infra([infra_col(Infrastructure.unit)]).dropna().squeeze().tolist()
    activity_per_tech_infra = read_infra([infra_col(Infrastructure.activity)]).dropna().squeeze().tolist()

    inv_cost_infra = read_infra(infra_year_cols(Infrastructure.investment)).fillna(0).to_numpy()
    fom_cost_infra = read_infra(infra_year_cols(Infrastructure.fixed_om)).fillna(0).to_numpy()
    ec_lifetime_infra = read_infra([infra_col(Infrastructure.ec_lifetime)]).fillna(0).squeeze().to_numpy()
    cap2act_infra = read_infra([infra_col(Infrastructure.cap2act)]).fillna(0).squeeze().to_numpy()
    tech_stock_exist_infra = read_infra([infra_col(Infrastructure.tech_stock_exist)]).fillna(0).squeeze().to_numpy()

    # Store in dictionary
    technologies['infra'] = {
        'ids': tech_infra,
        'categories': tech_categories_infra,
        'names': tech_names_infra,
        'units': tech_units_infra,
        'activity': activity_per_tech_infra,
        'costs': {
            'investments': inv_cost_infra,
            'foms': fom_cost_infra,
            'lifetimes': ec_lifetime_infra
        },
        'cap2acts': cap2act_infra,
        'stocks' : {
            'initial': tech_stock_exist_infra,
        }
    }

    print('--Saving infrastructure to duckdb')
    try:
        infrastructure_df = pd.DataFrame({
            "id": tech_infra,
            "category": tech_categories_infra,
            "name": tech_names_infra,
            "unit": tech_units_infra,
            "activity": activity_per_tech_infra,
            "cap2act": cap2act_infra,
            "lifetime": ec_lifetime_infra,
            "stock_initial": tech_stock_exist_infra,
        })
        con.execute(build_create_table_sql(
            "infrastructure", infrastructure_df, pk="id",
            fks=[("activity", "activities", "Name")]
        ))
        con.execute("INSERT INTO infrastructure SELECT * FROM infrastructure_df")

        infra_costs_idx = pd.MultiIndex.from_product([tech_infra, periods], names=["infra_id", "period"])
        infrastructure_costs_df = pd.DataFrame({
            "investment": inv_cost_infra.reshape(-1),
            "fom": fom_cost_infra.reshape(-1),
        }, index=infra_costs_idx).reset_index()
        con.execute(build_create_table_sql(
            "infrastructure_costs", infrastructure_costs_df, pk=["infra_id", "period"],
            fks=[("infra_id", "infrastructure", "id"), ("period", "periods", "period")]
        ))
        con.execute("INSERT INTO infrastructure_costs SELECT * FROM infrastructure_costs_df")
    except Exception as e:
        print(f"error in saving infrastructure to duckdb: {e}")

    # === Energy balance sheet ===
    print('--Reading energy balance sheet')
    # The activity-balance columns in this sheet are confirmed to carry the exact same
    # names, in the same order, as the Activities sheet's Name column - select by that
    # name instead of trusting a hardcoded Excel letter range to line up positionally.
    # header=2 only consumes the activity-name row; the sheet also has an annotation row
    # and a units row (e.g. '[PJ]') below it before the data starts, so skiprows must
    # skip past those too (6 rows), not just the header rows.
    energy_balance_header = pd.read_excel(file_name, sheet_name='EnergyBalance', header=2, nrows=0).columns.tolist()
    energy_balance_data = pd.read_excel(
        file_name, sheet_name='EnergyBalance', header=None, skiprows=6, nrows=550,
        usecols=range(len(energy_balance_header))
    )
    energy_balance_data.columns = energy_balance_header
    activity_balances = energy_balance_data[activities_df["Name"].tolist()].fillna(0).to_numpy() # MODIFIED & CHECK: Range changed from O:FF to O:ET. The remaining columns are empty. Double-check which columns are included in energy balances.

    # Store in dictionary
    technologies['balancers']['activity_balances'] = activity_balances

    print('--Saving energy balance to duckdb')
    try:
        energy_balance_df = matrix_to_long_df(
            tech_balancers, "tech_id", activity_balances, activities_df["Name"].tolist(), "activity_name", "value"
        )
        con.execute(build_create_table_sql(
            "energy_balance", energy_balance_df, pk=["tech_id", "activity_name"],
            fks=[("tech_id", "technologies", "id"), ("activity_name", "activities", "Name")]
        ))
        con.execute("INSERT INTO energy_balance SELECT * FROM energy_balance_df")
    except Exception as e:
        print(f"error in saving energy balance to duckdb: {e}")

    # === Retrofitting sheet ===
    print('--Reading retrofitting sheet')
    retro_data = pd.read_excel(file_name, sheet_name='Retrofitting', skiprows=2, nrows=496)

    # Derive individual parameters
    coord_bin = retro_data[Retrofitting.enabled] == 1
    retrofittings_cell = retro_data[coord_bin][[
        Retrofitting.tech_id_original, Retrofitting.tech_id_new, Retrofitting.investment_cost
    ]].to_numpy()
    retrofits_from = retrofittings_cell[:, 0].tolist()
    retrofits_to = retrofittings_cell[:, 1].tolist()
    retrofits_costs = retrofittings_cell[:, 2].tolist()

    # Store in dictionary
    technologies['retrofittings'] = {
        'to': retrofits_to,
        'from': retrofits_from,
        'costs': retrofits_costs
    }

    print('--Saving retrofittings to duckdb')
    try:
        retrofittings_df = pd.DataFrame({
            "from_tech": retrofits_from,
            "to_tech": retrofits_to,
            "cost": retrofits_costs,
        })
        valid_tech_ids = set(tech_balancers)
        valid_mask = retrofittings_df["from_tech"].isin(valid_tech_ids) & retrofittings_df["to_tech"].isin(valid_tech_ids)
        if (~valid_mask).any():
            print(f"warning: skipping {int((~valid_mask).sum())} retrofitting row(s) referencing tech ids not present in the Technologies sheet: "
                  f"{retrofittings_df.loc[~valid_mask, ['from_tech', 'to_tech']].to_dict('records')}")
        retrofittings_df = retrofittings_df[valid_mask]
        con.execute(build_create_table_sql(
            "retrofittings", retrofittings_df, pk=["from_tech", "to_tech"],
            fks=[("from_tech", "technologies", "id"), ("to_tech", "technologies", "id")]
        ))
        con.execute("INSERT INTO retrofittings SELECT * FROM retrofittings_df")
    except Exception as e:
        print(f"error in saving retrofittings to duckdb: {e}")



    # Policies sheet
    print('--Reading policies sheet')
    policies_data = pd.read_excel(file_name, sheet_name='Policies', header=None).to_numpy()

    # This sheet stacks three blocks (Taxes/Feedins/Subsidies) vertically rather than
    # laying fields out in named columns, so it's read positionally; the marker text
    # 'Units' in POLICY_COL_UNIT is what locates each block's start.
    POLICY_COL_ACTIVITY = 0
    POLICY_COL_UNIT = 1
    POLICY_COL_VALUES_START = 2
    policy_values_end = POLICY_COL_VALUES_START + len(periods)

    # Derive individual parameters
    units_indices = np.where(policies_data[:, POLICY_COL_UNIT] == 'Units')[0]
    taxes_data = policies_data[units_indices[0] + 1:units_indices[1], :]
    feedin_data = policies_data[units_indices[1] + 1:units_indices[2], :]
    subsidy_data = policies_data[units_indices[2] + 1:, :]

    # Store in dictionary
    policies = {
        'taxes': {
            'activities': taxes_data[:, POLICY_COL_ACTIVITY].tolist(),
            'values': taxes_data[:, POLICY_COL_VALUES_START:policy_values_end].astype(float)
        },
        'feedins': {
            'activities': feedin_data[:, POLICY_COL_ACTIVITY].tolist(),
            'values': feedin_data[:, POLICY_COL_VALUES_START:policy_values_end].astype(float)
        },
        'subsidies': {
            'activities': subsidy_data[:, POLICY_COL_ACTIVITY].tolist(),
            'values': subsidy_data[:, POLICY_COL_VALUES_START:policy_values_end].astype(float)
        }
    }

    print('--Saving policies to duckdb')
    try:
        for table_name, data in [
            ("policy_taxes", taxes_data),
            ("policy_feedins", feedin_data),
            ("policy_subsidies", subsidy_data),
        ]:
            policy_df = build_policy_long_df(data, periods)
            con.execute(build_create_table_sql(
                table_name, policy_df, pk=["activity_name", "period"],
                fks=[("activity_name", "activities", "Name"), ("period", "periods", "period")]
            ))
            con.execute(f"INSERT INTO {table_name} SELECT * FROM policy_df")
    except Exception as e:
        print(f"error in saving policies to duckdb: {e}")

    con.close()

    # === Save data to a file ===

    # Note: The commented out part is to save to a .mat file (if we want to ensure compatibility with matlab)
    # sio.savemat('data.mat', {
    #     'parameters': parameters,
    #     'types': types,
    #     'activities': activities,
    #     'profiles': profiles,
    #     'technologies': technologies,
    #     'agents': agents,
    #     'policies': policies
    # })
    # print("Data successfully saved to data.mat")
   
    # Save the database to a .pkl file (better if code is fully implemented in python)
    with open('data.pkl', 'wb') as file:
        pickle.dump({
            'parameters': parameters,
            'types': types,
            'activities': activities,
            'profiles': profiles,
            'technologies': technologies,
            'agents': agents,
            'policies': policies
        }, file)

    print("Data successfully saved to data.pkl")

    return parameters, types, activities, profiles, technologies, agents, policies

