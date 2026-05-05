# File to read data from the excel input file
import pandas as pd
import numpy as np
import pickle
# FIX: To suppress warning "UserWarning: Data Validation extension is not supported and will be removed warn(msg)" - not sure what to do with this, maybe fix later
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

def mod0_read_data(file_name):

    # Create empty dictionaries to store data that is derived from excel sheets
    parameters, types, activities, profiles, technologies, agents, policies = {}, {}, {}, {}, {}, {}, {}

    # === Parameters sheet ===
    print('--Reading parameters sheet')
    parameters_values = pd.read_excel(file_name, sheet_name='Parameters', usecols="B", skiprows=2, nrows=20).squeeze()

    # Extract individual parameters
    powinv_SPBT_benchmark = parameters_values.iloc[8]
    powinv_SPBT_min = parameters_values.iloc[9]
    powinv_CR_threshold = parameters_values.iloc[10]
    powinv_CR_min = parameters_values.iloc[11]
    powinv_NUF_threshold = parameters_values.iloc[12]
    powinv_NUF_min = parameters_values.iloc[13]
    scarcity_penalization = parameters_values.iloc[14]
    gas_premium = parameters_values.iloc[15]
    voll = parameters_values.iloc[16] / 3.6
    min_spread = parameters_values.iloc[17] / 3.6 # MODIFIED: min_spread row changed from 16 to 17 (mistake in Manuel's code)
    # min_spread = parameters_values.iloc[16] / 3.6
    gov_dr = parameters_values.iloc[18]
    exports_value = parameters_values.iloc[19]

    # Store parameters in dictionary
    parameters['powinv'] = {
        'SPBT_benchmark': powinv_SPBT_benchmark,
        'SPBT_min': powinv_SPBT_min,
        'CR_threshold': powinv_CR_threshold,
        'CR_min': powinv_CR_min,
        'NUF_threshold': powinv_NUF_threshold,
        'NUF_min': powinv_NUF_min,
    }
    parameters['scarcity'] = {
        'penalization' : scarcity_penalization,
        'gas_premium' : gas_premium
    } 
    parameters['voll'] = voll
    parameters['min_spread'] = min_spread
    parameters['gov_dr'] = gov_dr
    parameters['exports_value'] = exports_value

    # === Types sheet ===
    print('--Reading types sheet')
    activity_type = pd.read_excel(file_name, sheet_name='Types', usecols="B", skiprows=2, nrows=4).squeeze().tolist()
    sectors = pd.read_excel(file_name, sheet_name='Types', usecols="F", skiprows=2, nrows=27).dropna().squeeze().tolist()
    energy_labels = pd.read_excel(file_name, sheet_name='Types', usecols="K", skiprows=2, nrows=17, dtype=str).squeeze().tolist()
    energy_price_init = pd.read_excel(file_name, sheet_name='Types', usecols="M", skiprows=2, nrows=27).dropna().squeeze().to_numpy()

    # Store in dictionary
    types['activities'] = activity_type
    types['sectors'] = sectors
    types['energy'] = {
        'labels' : energy_labels,
        'price init' : energy_price_init,
    }

    # === Activities sheet ===
    print('--Reading activities sheet')
    activities_names = pd.read_excel(file_name, sheet_name='Activities', usecols="A", skiprows=7, nrows=192).dropna().squeeze().tolist()
    periods = pd.read_excel(file_name, sheet_name='Activities', usecols="C:I", skiprows = 6, nrows=1).squeeze().to_numpy()
    activities_net_volumes = pd.read_excel(file_name, sheet_name='Activities', usecols="C:I", skiprows=7, nrows=len(activities_names)).fillna(0).to_numpy() # CHECK: Matlab code uses len(periods) for activities_netVolumes, maybe need to adjust later
    activity_resolution = pd.read_excel(file_name, sheet_name='Activities', usecols="K", skiprows=7, nrows=len(activities_names)).squeeze().tolist()
    activity_type_act = pd.read_excel(file_name, sheet_name='Activities', usecols="L", skiprows=7, nrows=len(activities_names)).dropna().squeeze().tolist()
    activity_label = pd.read_excel(file_name, sheet_name='Activities', usecols="O", skiprows=7, nrows=len(activities_names)).squeeze().tolist()
    activity_agent = pd.read_excel(file_name, sheet_name='Activities', usecols="P", skiprows=7, nrows=len(activities_names)).squeeze().tolist()

    # Store in dictionary
    activities['names'] = activities_names
    activities['periods'] = periods
    activities['volumes'] = activities_net_volumes
    activities['resolution'] = activity_resolution
    activities['types'] = activity_type_act
    activities['labels'] = activity_label
    activities['agents'] = activity_agent
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

    # === Agents sheet ===
    print('--Reading agents sheet')
    agent_profiles = pd.read_excel(file_name, sheet_name='Agents', usecols="A", skiprows=3, nrows=5).dropna().squeeze().tolist()
    agent_types = pd.read_excel(file_name, sheet_name='Agents', usecols="C:F", skiprows=1, nrows=1).dropna().squeeze().tolist()
    multiCriteria_categories = pd.read_excel(file_name, sheet_name='Agents', usecols="A", skiprows=9, nrows=4).dropna().squeeze().tolist()
    agents_dr = pd.read_excel(file_name, sheet_name='Agents', usecols="B", skiprows=3, nrows=5).fillna(0).squeeze().to_numpy() / 100
    agents_populations = pd.read_excel(file_name, sheet_name='Agents', usecols="C:F", skiprows=3, nrows=5).fillna(0).to_numpy() / 100
    weights_multiCriteria = pd.read_excel(file_name, sheet_name='Agents', usecols="C:F", skiprows=9, nrows=4).fillna(0).to_numpy()

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

