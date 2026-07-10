# File to read data from the excel input file
from typing import *

import pandas as pd
import numpy as np
import pickle
import duckdb
from Constants import Parameters
import re
# FIX: To suppress warning "UserWarning: Data Validation extension is not supported and will be removed warn(msg)" - not sure what to do with this, maybe fix later
import warnings

from Constants.Parameters import Activities, Agents

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

def dtype_to_sql(dtype):
    if "int" in str(dtype):
        return "INTEGER"
    elif "float" in str(dtype):
        return "DOUBLE"
    else:
        return "VARCHAR"

def mod0_read_data_save_duck(file_name):

    # Create empty dictionaries to store data that is derived from excel sheets
    parameters, types, activities, profiles, technologies, agents, policies = {}, {}, {}, {}, {}, {}, {}

    con = duckdb.connect("SIMmodel.duckdb")
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
    # The only one where I kept this way of reading per coumn and excel column name
    # TO DO: Make this reading nicer
    print('--Reading types sheet')
    activity_type = pd.read_excel(file_name, sheet_name='Types', usecols="B", skiprows=2, nrows=4).squeeze().tolist()
    sectors = pd.read_excel(file_name, sheet_name='Types', usecols="F", skiprows=2, nrows=27).dropna().squeeze().tolist()
    energy_labels = pd.read_excel(file_name, sheet_name='Types', usecols="K", skiprows=2, nrows=17, dtype=str, keep_default_na=False).squeeze().tolist()
    energy_price_init = pd.read_excel(file_name, sheet_name='Types', usecols="M", skiprows=2, nrows=27).dropna().squeeze().to_numpy()

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
    agents.columns = [
        " / ".join(str(x) for x in col if pd.notna(x) and not str(x).startswith("Unnamed"))
        for col in agents.columns
    ]
    agents.columns = [re.sub(r'\s*/?\s*\[.*\]\s*$', '', col).strip() for col in agents.columns]
    agents = agents.rename(columns={Agents.rates: "rates"})
    agents = agents.rename(columns={Agents.profiles: "Name"})
    # 1. select columns that start with the prefix
    selected_cols = [col for col in agents.columns if str(col).startswith(Agents.types)]
    # 2. extract the agent types
    types = [str(col)[len(Agents.types):].strip(" /").strip() for col in selected_cols]
    types_df = pd.DataFrame(types, columns=["Name"])

    rename_map = dict(zip(selected_cols, types))
    agents = agents.rename(columns=rename_map)

    agent_profiles = agents[agents["rates"].notna()]
    agent_profiles["rates"] = agent_profiles["rates"] / 100
    agent_profiles_toTable = agent_profiles[["Name", "rates"]]
    population_df = agent_profiles.melt(
        id_vars=["Name"],  # FK to agent_profiles
        value_vars=types,  # the agent-type columns: Innovators, Early adopters, Majority, Laggards
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
        value_vars=types,  # the agent-type columns: Innovators, Early adopters, Majority, Laggards
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

    agent_profiles = pd.read_excel(file_name, sheet_name='Agents', usecols="A", skiprows=3,
                                   nrows=5).dropna().squeeze().tolist()
    agent_types = pd.read_excel(file_name, sheet_name='Agents', usecols="C:F", skiprows=1,
                                nrows=1).dropna().squeeze().tolist()
    multiCriteria_categories = pd.read_excel(file_name, sheet_name='Agents', usecols="A", skiprows=9,
                                             nrows=4).dropna().squeeze().tolist()
    agents_dr = pd.read_excel(file_name, sheet_name='Agents', usecols="B", skiprows=3, nrows=5).fillna(
        0).squeeze().to_numpy() / 100
    agents_populations = pd.read_excel(file_name, sheet_name='Agents', usecols="C:F", skiprows=3, nrows=5).fillna(
        0).to_numpy() / 100
    weights_multiCriteria = pd.read_excel(file_name, sheet_name='Agents', usecols="C:F", skiprows=9, nrows=4).fillna(
        0).to_numpy()

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
    activities = pd.read_excel(
        file_name,
        sheet_name='Activities',
        skiprows=6,
        header=[0, 1],
        keep_default_na=False,
    )
    activities.columns = [
        " / ".join(str(x) for x in col if pd.notna(x) and not str(x).startswith("Unnamed"))
        for col in activities.columns
    ]
    activities.columns = [re.sub(r'\s*/?\s*\[.*\]\s*$', '', col).strip() for col in activities.columns]

    prefix = Parameters.Activities.periods_start

    # 1. select columns that start with the prefix
    selected_cols = [col for col in activities.columns if str(col).startswith(prefix)]

    # 2. extract the years from those column names, and cast to int
    periods = [int(re.search(r'\d+', str(col)[len(prefix):]).group()) for col in selected_cols]

    print(selected_cols)
    print(periods)
    activities.columns = [rename_volumes_col(col) for col in activities.columns]

    activities = activities.rename(columns={Activities.activities_names: "Name"})
    activities = activities.rename(columns={Activities.activity_resolution: "activity_resolution"})
    activities = activities.rename(columns={Activities.activity_type_act: "activity_type"})
    activities = activities.rename(columns={Activities.activity_label: "energy_label"})
    activities = activities.rename(columns={Activities.activity_agent: "agent_profile"})
    activities_df=activities

    try:
        cols_sql = ",\n    ".join(f'"{col}" {dtype_to_sql(dt)}' for col, dt in activities.dtypes.items())

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
        con.close()
    except:
        print("error in saving activities to duckdb")

    activities['drivers'] = {}
    activities['energies'] = {}
    activities['emissions'] = {}
    activities['electricity'] = {}
    activities['gaseous'] = {}
    activities['infra'] = {}
    activities['prices'] = {}

    # === Hourly profiles sheet ===
    print('--Reading hourly profiles sheet')
    profile_types = pd.read_excel(file_name, sheet_name='HourlyProfiles', usecols="D:BB", skiprows=1, nrows=1).dropna(axis=1).squeeze().tolist()
    hourly_profiles = pd.read_excel(file_name, sheet_name='HourlyProfiles', usecols="D:BB", skiprows=3, nrows=8760).dropna(axis=1).to_numpy()

    # Store in dictionary
    profiles['types'] = profile_types
    profiles['shapes'] = hourly_profiles

    # === Price profiles sheet ===
    print('--Reading price profiles sheet')
    interconnector_raw = pd.read_excel(file_name, sheet_name='PriceProfiles', usecols="D:J", nrows=1).dropna(axis=1).squeeze().tolist() # MODIFIED: Changed rows to D:J since the price profile sheet was empty otherwise
    price_profiles_raw = pd.read_excel(file_name, sheet_name='PriceProfiles', usecols="D:J", skiprows=3, nrows=8760).to_numpy()
    nIC = len(interconnector_raw) // len(periods)

    price_profiles = np.zeros((8760, nIC, len(periods)))
    interconnector = []
    for i in range(nIC):
        interconnector.append(interconnector_raw[len(periods) * i])
        price_profiles[:, i, :] = price_profiles_raw[:, len(periods) * i:len(periods) * (i + 1)]

    # Store in dictionary
    profiles['interconnectors'] = interconnector
    profiles['prices'] = price_profiles

    # === Technologies sheet ===
    print('--Reading technologies sheet')
    tech_balancers = pd.read_excel(file_name, sheet_name='Technologies', usecols="A", skiprows=5, nrows=793).dropna().squeeze().tolist()
    tech_names = pd.read_excel(file_name, sheet_name='Technologies', usecols="F", skiprows=5, nrows=793).dropna().squeeze().tolist()
    tech_sector = pd.read_excel(file_name, sheet_name='Technologies', usecols="C", skiprows=5, nrows=793).dropna().squeeze().tolist()
    tech_subsector = pd.read_excel(file_name, sheet_name='Technologies', usecols="D", skiprows=5, nrows=793).dropna().squeeze().tolist()
    tech_units = pd.read_excel(file_name, sheet_name='Technologies', usecols="G", skiprows=5, nrows=793).dropna().squeeze().tolist()
    activity_per_tech = pd.read_excel(file_name, sheet_name='Technologies', usecols="E", skiprows=5, nrows=793).dropna().squeeze().tolist()
    tech_categories = pd.read_excel(file_name, sheet_name='Technologies', usecols="B", skiprows=5, nrows=793).dropna().squeeze().tolist()

    inv_cost = pd.read_excel(file_name, sheet_name='Technologies', usecols="H:N", skiprows=5, nrows=793).fillna(0).to_numpy()
    fom_cost = pd.read_excel(file_name, sheet_name='Technologies', usecols="P:V", skiprows=5, nrows=793).fillna(0).to_numpy()
    vom_cost = pd.read_excel(file_name, sheet_name='Technologies', usecols="W:AC", skiprows=5, nrows=793).fillna(0).to_numpy()

    ec_lifetime = pd.read_excel(file_name, sheet_name='Technologies', usecols="AF", skiprows=5, nrows=793).fillna(0).squeeze().to_numpy()
    cap2act = pd.read_excel(file_name, sheet_name='Technologies', usecols="AH", skiprows=5, nrows=793).fillna(0).squeeze().to_numpy()

    dispatch_type_tech = pd.read_excel(file_name, sheet_name='Technologies', usecols="AI", skiprows=5, nrows=793).dropna().squeeze().tolist()
    hourly_profile_tech = pd.read_excel(file_name, sheet_name='Technologies', usecols="AJ", skiprows=5, nrows=793).dropna().squeeze().tolist()
    social_perception_tech = pd.read_excel(file_name, sheet_name='Technologies', usecols="AL", skiprows=5, nrows=793).dropna().squeeze().tolist()
    perceived_complexity_tech = pd.read_excel(file_name, sheet_name='Technologies', usecols="AM", skiprows=5, nrows=793).dropna().squeeze().tolist()

    subsidy_subject = pd.read_excel(file_name, sheet_name='Technologies', usecols="AN", skiprows=5, nrows=793).fillna(0).squeeze().to_numpy().astype(bool)
    feedin_subject = pd.read_excel(file_name, sheet_name='Technologies', usecols="AO", skiprows=5, nrows=793).fillna(0).squeeze().to_numpy().astype(bool)

    shedding_capacity = pd.read_excel(file_name, sheet_name='Technologies', usecols="AV", skiprows=5, nrows=793).fillna(0).squeeze().to_numpy()
    shedding_limits = pd.read_excel(file_name, sheet_name='Technologies', usecols="AW", skiprows=5, nrows=793).fillna(0).squeeze().to_numpy()
    shedding_guarantee = pd.read_excel(file_name, sheet_name='Technologies', usecols="AY", skiprows=5, nrows=793).fillna(0).squeeze().to_numpy()

    flexibility_form = pd.read_excel(file_name, sheet_name='Technologies', usecols="BC", skiprows=5, nrows=793).dropna().squeeze().tolist()
    flexibility_activity = pd.read_excel(file_name, sheet_name='Technologies', usecols="BD", skiprows=5, nrows=793).dropna().squeeze().tolist() # CHECK: flexibility_activity only retrieves 38 values and skips all empty rows. I don't know if that's a problem - compare with matlab
    flexibility_capacity = pd.read_excel(file_name, sheet_name='Technologies', usecols="BE", skiprows=5, nrows=793).fillna(0).squeeze().to_numpy()
    flexibility_volume = pd.read_excel(file_name, sheet_name='Technologies', usecols="BF", skiprows=5, nrows=793).fillna(0).squeeze().to_numpy()
    flexibility_range = pd.read_excel(file_name, sheet_name='Technologies', usecols="BG", skiprows=5, nrows=793).fillna(0).squeeze().tolist()
    flexibility_losses = pd.read_excel(file_name, sheet_name='Technologies', usecols="BH", skiprows=5, nrows=793).fillna(0).squeeze().to_numpy()
    flexibility_nonnegotiable = pd.read_excel(file_name, sheet_name='Technologies', usecols="BI", skiprows=5, nrows=793).fillna(0).squeeze().to_numpy()

    buffer_up = pd.read_excel(file_name, sheet_name='Technologies', usecols="BM", skiprows=5, nrows=793).fillna(0).squeeze().to_numpy()
    buffer_down = pd.read_excel(file_name, sheet_name='Technologies', usecols="BN", skiprows=5, nrows=793).fillna(0).squeeze().to_numpy()
    buffer_capacity = pd.read_excel(file_name, sheet_name='Technologies', usecols="BO", skiprows=5, nrows=793).fillna(0).squeeze().to_numpy()

    tech_stock_deploy = pd.read_excel(file_name, sheet_name='Technologies', usecols="BR", skiprows=5, nrows=793).fillna(0).squeeze().to_numpy()
    tech_stock_exist = pd.read_excel(file_name, sheet_name='Technologies', usecols="BS", skiprows=5, nrows=793).fillna(0).squeeze().to_numpy()
    # Derive tech_stock_dec in correct format
    nTb = len(tech_balancers)
    nP = len(periods)
    raw_data = pd.read_excel(file_name, sheet_name='Technologies', usecols="BT:BY", skiprows=5, nrows=793).fillna(0).to_numpy()
    tech_stock_dec = np.zeros((nTb, nP))
    tech_stock_dec[:, 1:] = raw_data
    tech_stock_min = pd.read_excel(file_name, sheet_name='Technologies', usecols="BZ:CF", skiprows=5, nrows=793).fillna(0).to_numpy()
    tech_stock_max = pd.read_excel(file_name, sheet_name='Technologies', usecols="CG:CM", skiprows=5, nrows=793).fillna(0).to_numpy()

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

    # === Infrastructure sheet ===
    print('--Reading infrastructure sheet')
    tech_infra = pd.read_excel(file_name, sheet_name='Infrastructure', usecols="A", skiprows=4, nrows=15).dropna().squeeze().tolist()
    tech_categories_infra = pd.read_excel(file_name, sheet_name='Infrastructure', usecols="B", skiprows=4, nrows=15).dropna().squeeze().tolist()
    tech_names_infra = pd.read_excel(file_name, sheet_name='Infrastructure', usecols="E", skiprows=4, nrows=15).dropna().squeeze().tolist()
    tech_units_infra = pd.read_excel(file_name, sheet_name='Infrastructure', usecols="F", skiprows=4, nrows=15).dropna().squeeze().tolist()
    activity_per_tech_infra = pd.read_excel(file_name, sheet_name='Infrastructure', usecols="AA", skiprows=4, nrows=15).dropna().squeeze().tolist()

    inv_cost_infra = pd.read_excel(file_name, sheet_name='Infrastructure', usecols="G:M", skiprows=4, nrows=15).fillna(0).to_numpy()
    fom_cost_infra = pd.read_excel(file_name, sheet_name='Infrastructure', usecols="O:U", skiprows=4, nrows=15).fillna(0).to_numpy()
    ec_lifetime_infra = pd.read_excel(file_name, sheet_name='Infrastructure', usecols="W", skiprows=4, nrows=15).fillna(0).squeeze().to_numpy()
    cap2act_infra = pd.read_excel(file_name, sheet_name='Infrastructure', usecols="Y", skiprows=4, nrows=15).fillna(0).squeeze().to_numpy()
    tech_stock_exist_infra = pd.read_excel(file_name, sheet_name='Infrastructure', usecols="AE", skiprows=4, nrows=15).fillna(0).squeeze().to_numpy()

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

    # === Energy balance sheet ===
    print('--Reading energy balance sheet')
    activity_balances = pd.read_excel(file_name, sheet_name='EnergyBalance', usecols="O:ET", skiprows=5, nrows=550).fillna(0).to_numpy() # MODIFIED & CHECK: Range changed from O:FF to O:ET. The remaining columns are empty. Double-check which columns are included in energy balances.

    # Store in dictionary
    technologies['balancers']['activity_balances'] = activity_balances

    # === Retrofitting sheet ===
    print('--Reading retrofitting sheet')
    retro_data = pd.read_excel(file_name, sheet_name='Retrofitting', usecols="A:G", skiprows=2, nrows=496)

    # Derive individual parameters
    coord_bin = retro_data.iloc[:, 5] == 1
    retrofittings_cell = retro_data[coord_bin].iloc[:, [0, 1, 6]].to_numpy()
    retrofits_from = retrofittings_cell[:, 0].tolist()
    retrofits_to = retrofittings_cell[:, 1].tolist()
    retrofits_costs = retrofittings_cell[:, 2].tolist()

    # Store in dictionary
    technologies['retrofittings'] = {
        'to': retrofits_to,
        'from': retrofits_from,
        'costs': retrofits_costs
    }



    # Policies sheet
    print('--Reading policies sheet')
    policies_data = pd.read_excel(file_name, sheet_name='Policies', header=None).to_numpy()

    # Derive individual parameters
    units_indices = np.where(policies_data[:, 1] == 'Units')[0]
    taxes_data = policies_data[units_indices[0] + 1:units_indices[1], :]
    feedin_data = policies_data[units_indices[1] + 1:units_indices[2], :]
    subsidy_data = policies_data[units_indices[2] + 1:, :]

    # Store in dictionary
    policies = {
        'taxes': {
            'activities': taxes_data[:, 0].tolist(),
            'values': taxes_data[:, 2:9].astype(float)
        },
        'feedins': {
            'activities': feedin_data[:, 0].tolist(),
            'values': feedin_data[:, 2:9].astype(float)
        },
        'subsidies': {
            'activities': subsidy_data[:, 0].tolist(),
            'values': subsidy_data[:, 2:9].astype(float)
        }
    }

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

