import numpy as np

def disp_initialize_power(dimensions, activities, technologies, profiles, tech_use_hourly, iP):
    # Extract Parameters
    nH = dimensions['nH']
    nA = dimensions['nA']
    nAk = dimensions['nAk']
    activities_names = activities['names']
    activities_elec = activities['electricity']['names']
    activity_per_tech = technologies['balancers']['activities']
    activity_balances = technologies['balancers']['activity_balances']
    tech_stock = technologies['balancers']['stocks']['evolution'][:, iP]
    tech_subsector = technologies['balancers']['subsectors']
    cap2act = technologies['balancers']['cap2acts']
    vom_cost = technologies['balancers']['costs']['voms'][:, iP]
    shedding_capacity = technologies['balancers']['shedding']['capacity']
    shedding_limits = technologies['balancers']['shedding']['limits']
    shedding_guarantee = technologies['balancers']['shedding']['guarantee']
    flexibility_form = technologies['balancers']['flexibility']['form']
    flexibility_capacity = technologies['balancers']['flexibility']['capacity']
    flexibility_volume = technologies['balancers']['flexibility']['volume']
    flexibility_losses = technologies['balancers']['flexibility']['losses']
    flexibility_nonnegotiable = technologies['balancers']['flexibility']['nonnegotiable']
    flexibility_range_days = technologies['balancers']['flexibility']['range_days']
    hourly_profile_tech = technologies['balancers']['profiles']
    profile_type = profiles['types']
    hourly_profiles = profiles['shapes']
    interconnector = profiles['interconnectors']
    price_profiles = profiles['prices'][:, :, iP]

    # Identify generators, shedders, and interconnectors
    tech_generators_coord = np.array(
        (np.array([subsector == 'Inland generation' for subsector in tech_subsector], dtype=bool) |
        np.array([subsector == 'Generation' for subsector in tech_subsector], dtype=bool) |
        np.array([subsector == 'Undispatched' for subsector in tech_subsector], dtype=bool))
    )
    tech_shedding_coord = np.array((shedding_capacity > 0) & (tech_stock > 0), dtype=bool)
    tech_loadshifts_coord = np.logical_and(np.array(flexibility_form) == 'DR shifting', tech_stock > 0)
    tech_batteries_coord = np.logical_and(np.array(flexibility_form) == 'Storage', tech_stock > 0)
    tech_interconnectors_coord = np.array(tech_subsector) == 'XC Trade'
    nG = np.sum(tech_generators_coord)

    # Preallocate arrays
    gen_per_elec = np.zeros((nG, nAk), dtype=bool)
    gen_xc_costs_hourly = np.zeros((nH, nG))
    gen_availability_hourly = np.zeros((nH, nG))
    gen_balance_hourly = np.zeros((nH, nA, nG))
    elec_demand_hourly = np.zeros((nH, nAk))
    elec_cogenerated = np.zeros((nAk, 1))

    

    
    # Build the hourly demand profiles of electricity
    for iAk in range(nAk):
        coord_act = np.array([activity == activities_elec[iAk] for activity in activities_names], dtype=bool)

        coord_tech = (
            (activity_balances[:, coord_act] != 0).any(axis=1).astype(int) *  # Ensures a column vector
            (1 - tech_shedding_coord.astype(int)) *
            (1 - tech_generators_coord.astype(int)) *
            (1 - tech_loadshifts_coord.astype(int)) *
            (1 - tech_batteries_coord.astype(int)) *
            (1 - tech_interconnectors_coord.astype(int))
        ).astype(bool)
        coord_tech = coord_tech.reshape(-1, 1)

        tech_sel = np.where(coord_tech)[0]


        nT = np.sum(coord_tech)
        for iT in range(nT):
            coord_itb = tech_sel[iT]
            elec_demand_hourly[:, iAk] -= tech_use_hourly[:, coord_itb] * activity_balances[coord_itb, coord_act]

            # debugging
            # val = np.sum(tech_use_hourly[:, coord_itb])
            # bal = activity_balances[coord_itb, coord_act]
            # print("Shapes:", val, bal.shape)
            # print("iT:", iT, "Values:", val, bal)
            # print("iT:", iT, "Values:", val)
            # If bal is bigger than length 1, or 0 in length, see if that matches MATLAB
            # print(f"  iT={iT}, coord_itb={coord_itb}, bal.shape={bal.shape}, bal={bal}, val={val}")
            # add_amount = val * bal
            # check sign:
            # print(f"    => product = {add_amount}")
            # Then the max(0, ...) step
            # to_add = max(0, add_amount)  # If add_amount is an array, this may fail or do the wrong thing
            # elec_cogenerated[iAk, 0] += to_add
            
            elec_cogenerated[iAk, 0] += max(0, np.sum(tech_use_hourly[:, coord_itb]) * activity_balances[coord_itb, coord_act])
            

    # Get the generators descriptions
    gen_vom = vom_cost[tech_generators_coord]
    tech_subsector = np.array(tech_subsector)
    generators_subsector = tech_subsector[tech_generators_coord]
    activity_balance_gen = activity_balances[tech_generators_coord, :]
    activity_per_tech = np.array(activity_per_tech)
    generators_electricity = activity_per_tech[tech_generators_coord]
    generators_activity = tech_stock[tech_generators_coord] * cap2act[tech_generators_coord]
    hourly_profile_tech = np.array(hourly_profile_tech)
    availability_profiles_generators = hourly_profile_tech[tech_generators_coord]

    for iG in range(nG):
        # Identify the electricity activity per generator
        iA = np.array([name == generators_electricity[iG] for name in activities_names])
        iAk = np.array([name == generators_electricity[iG] for name in activities_elec])
        gen_per_elec[iG, iAk] = True
        activity_balance_gen[iG, iA] = 0

        # Get the hourly availability profiles of the generators
        coord_profile = np.array([profile == availability_profiles_generators[iG] for profile in profile_type], dtype=bool)
        gen_availability_hourly[:, iG] = hourly_profiles[:, coord_profile].squeeze() * generators_activity[iG]

        # Get the hourly variable costs of the generators
        if generators_subsector[iG] == 'Inland generation':  # Interconnectors
            elec_generated = activities_elec[iAk]
            if np.sum(interconnector == elec_generated) > 0:
                coord_IC = (interconnector == elec_generated)
                gen_xc_costs_hourly[:, iG] = price_profiles[:, coord_IC]

        # Obtain the hourly activity balances
        gen_balance_hourly[:, :, iG] = np.ones((nH, 1)) * activity_balance_gen[iG, :]

    # debugging
    # sum_per_generator = np.sum(gen_balance_hourly, axis=(0, 1))
    # print(sum_per_generator)

    gen_per_elec = gen_per_elec.astype(bool)

    # Shedding calculations
    nS = np.sum(tech_shedding_coord)
    shed_min_demand_hourly = np.zeros((nH, nS))
    shed_max_demand_hourly = np.zeros((nH, nS))
    shed_max_volume_hourly = np.zeros((nH, nS))
    shed_min_volume_hourly = np.zeros((nH, nS))
    shed_per_elec = np.zeros((nS, nAk), dtype=bool)
    shed_multiplier = np.zeros((nS, 1))

    shedding_guarantee_values = shedding_guarantee[tech_shedding_coord]
    shedding_limits_values = shedding_limits[tech_shedding_coord]
    shedding_capacity_values = shedding_capacity[tech_shedding_coord]

    operational_profiles_shedding = hourly_profile_tech[tech_shedding_coord]
    tech_stock_shedding = tech_stock[tech_shedding_coord]
    cap2act_shedding = cap2act[tech_shedding_coord]
    activity_balances_shed = activity_balances[tech_shedding_coord, :]

    for iS in range(nS):
        elec_balance = 0
        for iAk in range(nAk):
            coord_act = np.array([activity == activities_elec[iAk] for activity in activities_names], dtype=bool)
            act_use = -activity_balances_shed[iS, coord_act]
            if act_use > 0:
                shed_per_elec[iS, iAk] = True
                elec_balance = act_use

        shed_multiplier[iS, 0] = elec_balance

        coord_profile = np.array([profile == operational_profiles_shedding[iS] for profile in profile_type], dtype=bool)
        ref_profile = (tech_stock_shedding[iS] * elec_balance *
                    cap2act_shedding[iS] * hourly_profiles[:, coord_profile].squeeze())
        shed_potential = np.minimum(ref_profile * shedding_limits_values[iS],
                                    tech_stock_shedding[iS] * cap2act_shedding[iS] * shedding_capacity_values[iS])

        shed_max_demand_hourly[:, iS] = ref_profile
        shed_min_demand_hourly[:, iS] = ref_profile - shed_potential
        shed_max_volume_hourly[:, iS] = shed_max_demand_hourly[:, iS] / np.sum(ref_profile)
        shed_min_volume_hourly[:, iS] = shed_min_demand_hourly[:, iS] / np.sum(ref_profile)

    shed_per_elec = shed_per_elec.astype(bool)

    # Loadshifting calculations
    nL = np.sum(tech_loadshifts_coord)
    loadshifts_demand_hourly = np.zeros((nH, nL))
    loadshifts_per_elec = np.zeros((nL, nAk), dtype=bool)
    loadshifts_per_uoa = np.zeros((nL, 1))
    loadshifts_efficiencies = 0
    loadshifts_capacities = 0
    loadshifts_min = 0
    loadshifts_range = 0


    if nL > 0:
        loadshifts_efficiencies = 1 - flexibility_losses[tech_loadshifts_coord]
        loadshifts_capacities = flexibility_capacity[tech_loadshifts_coord] * tech_stock[tech_loadshifts_coord]
        loadshifts_min = flexibility_nonnegotiable[tech_loadshifts_coord]
        loadshifts_range = flexibility_range_days[tech_loadshifts_coord]

        operational_profiles_loadshifts = hourly_profile_tech[tech_loadshifts_coord]
        tech_stock_loadshifts = tech_stock[tech_loadshifts_coord]
        cap2act_loadshifts = cap2act[tech_loadshifts_coord]
        activity_balances_loadshifts = activity_balances[tech_loadshifts_coord, :]

        for iL in range(nL):
            elec_balance = 0
            for iAk in range(nAk):
                coord_act = np.array([activity == activities_elec[iAk] for activity in activities_names], dtype=bool)
                act_use = -activity_balances_loadshifts[iL, coord_act]
                if act_use > 0:
                    loadshifts_per_elec[iL, iAk] = True
                    elec_balance = act_use
                    loadshifts_per_uoa[iL, 0] = act_use

            coord_profile = np.array([profile == operational_profiles_loadshifts[iL] for profile in profile_type], dtype=bool)
            loadshifts_demand_hourly[:, iL] = (tech_stock_loadshifts[iL] * elec_balance *
                                            cap2act_loadshifts[iL] * hourly_profiles[:, coord_profile].squeeze())

    # Batteries calculations
    nB = np.sum(tech_batteries_coord)
    bat_per_elec = np.zeros((nB, nAk), dtype=bool)
    bat_efficiency = 0.5
    bat_capacity = 0
    bat_volume = 0
    bat_vom = 0
    bat_stock = 0

    if nB > 0:
        bat_efficiency = 1 - flexibility_losses[tech_batteries_coord]
        bat_capacity = flexibility_capacity[tech_batteries_coord]
        bat_volume = flexibility_volume[tech_batteries_coord]
        bat_vom = vom_cost[tech_batteries_coord]
        bat_stock = tech_stock[tech_batteries_coord]
        bat_act = activity_per_tech[tech_batteries_coord]

        bat_per_elec = np.zeros((nB, nAk))
        for iB in range(nB):
            iAk = np.array([activity == bat_act[iB] for activity in activities_elec], dtype=bool)
            bat_per_elec[iB, iAk] = 1

        bat_per_elec = bat_per_elec.astype(bool)

    

    # Obtain figures for the interconnectors
    nI = np.sum(tech_interconnectors_coord)

    # Preallocate our targets
    xc_efficiencies = np.zeros((nI, 1))
    xc_per_elec = np.zeros((nAk, nAk, nI), dtype=bool)
    xc_demand = np.zeros((nH, nI))
    xc_supply = np.zeros((nH, nI))

    # Interconnector VOM
    xc_vom = vom_cost[tech_interconnectors_coord]

    # Identify the to and froms and the efficiencies
    activity_balance_xc = activity_balances[tech_interconnectors_coord, :]
    interconnector_to = activity_per_tech[tech_interconnectors_coord]
    interconnector_profiles = hourly_profile_tech[tech_interconnectors_coord]
    techStock_interconnectors = tech_stock[tech_interconnectors_coord]
    # Manuel: there seems to be a mistake in the code, it says tech_stock also for cap2act interconnectors
    cap2act_interconnectors = tech_stock[tech_interconnectors_coord]

    for iI in range(nI):
        # Identify the to coordinate
        iAk_to = np.array([activity == interconnector_to[iI] for activity in activities_elec], dtype=bool)

        coord_act = (activity_balance_xc[iI, :] < 0)

        iAk_from = np.array([activity == activities_names[np.where(coord_act)[0][0]] for activity in activities_elec], dtype=bool)

        # Save the link
        xc_per_elec[iAk_to, iAk_from, iI] = True

        # Efficiency
        xc_efficiencies[iI, 0] = -1 / activity_balance_xc[iI, coord_act][0]

        # Get the hourly availability profiles of the XC technologies
        coord_profile = np.array([profile == interconnector_profiles[iI] for profile in profile_type], dtype=bool)
        ref_profile = hourly_profiles[:, coord_profile].squeeze()

        # Supply and demand profiles
        xc_demand[:, iI] = ref_profile * techStock_interconnectors[iI] * cap2act_interconnectors[iI]
        xc_supply[:, iI] = -xc_demand[:, iI] / xc_efficiencies[iI, 0]

    xc_per_elec = xc_per_elec.astype(bool)


    return (gen_vom, gen_balance_hourly, gen_availability_hourly, gen_xc_costs_hourly,
            gen_per_elec, elec_demand_hourly, shedding_guarantee_values, shed_max_volume_hourly,
            shed_min_volume_hourly, shed_per_elec, shed_max_demand_hourly,
            shed_min_demand_hourly, shed_multiplier, loadshifts_efficiencies, loadshifts_capacities, 
            loadshifts_min, loadshifts_per_uoa, loadshifts_range, loadshifts_per_elec,
            loadshifts_demand_hourly, bat_efficiency, bat_capacity, bat_volume, 
            bat_per_elec, bat_vom, bat_stock, 
            xc_efficiencies, xc_vom, xc_per_elec, xc_demand, xc_supply, elec_cogenerated,
            tech_generators_coord, tech_shedding_coord, 
            tech_loadshifts_coord, tech_batteries_coord, tech_interconnectors_coord)
