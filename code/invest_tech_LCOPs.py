import numpy as np

def invest_tech_LCOPs(dimensions, activities, technologies, policies, retrofit_potential, retrofit_cost, iP):
    
    # Extract parameters
    nAe = dimensions['nAe']
    nAk = dimensions['nAk']
    nTb = dimensions['nTb']
    nTL = dimensions['nTL']
    activities_names = activities['names']
    activity_type_act = activities['types']
    activities_energy = activities['energies']['names']
    activities_elec = activities['electricity']['names']
    
    if iP == 0:
        energy_prices = activities['energies']['prices']['initialized']
        emission_prices = activities['emissions']['prices']['initialized']
    else:
        energy_prices = activities['energies']['prices']['yearly'][:, iP - 1]
        emission_prices = activities['emissions']['prices']['yearly'][:, iP - 1]
        energy_prices_ranges = activities['energies']['prices']['ranges'][:, :, iP - 1]
    
    price_ranges = activities['energies']['prices']['price_ranges']
    activity_per_tech = technologies['balancers']['activities']
    activity_balances = technologies['balancers']['activity_balances']
    cap2act = technologies['balancers']['cap2acts']
    annuity_fact = technologies['balancers']['costs']['annuity']
    inv_cost = technologies['balancers']['costs']['investments'][:, iP]
    fom_cost = technologies['balancers']['costs']['foms'][:, iP]
    vom_cost = technologies['balancers']['costs']['voms'][:, iP]
    shedding_guarantee = technologies['balancers']['shedding']['guarantee']
    flexibility_form = technologies['balancers']['flexibility']['form']
    flexibility_nonnegotiable = technologies['balancers']['flexibility']['nonnegotiable']
    subsidy_subject = technologies['balancers']['policies']['subsidy_subject']
    subsidy_activities = policies['subsidies']['activities']
    subsidy_values = policies['subsidies']['values']
    feedin_subject = technologies['balancers']['policies']['feedin_subject']
    feedin_activities = policies['feedins']['activities']
    feedin_values = policies['feedins']['values']
    taxes_activities = policies['taxes']['activities']
    taxes_values = policies['taxes']['values']

    # Identify energy and emissions coordinates
    iAc = np.array(activity_type_act) == 'Emission'
    iAe = np.logical_or(np.array(activity_type_act) == 'Energy', np.array(activity_type_act) == 'Fix Energy')

    # Preallocate arrays
    tech_lcops = np.zeros(nTb)
    tech_lcops_matrix = np.zeros((nTb, nTL))

    for iTb in range(nTb):

        # Modify the vector included for the activity balance
        iA = np.array(activities_names) == activity_per_tech[iTb]
        technology_balance = activity_balances[iTb, :].copy()
        technology_balance[iA] = 0
        energy_balance = technology_balance[iAe]

        # Modify the energy price vector for shedding and flexible technologies
        used_energy_prices = energy_prices.copy()
        if ((shedding_guarantee[iTb] > 0) or (flexibility_form[iTb] == 'DR shifting')) and (iP > 0):

            # Find which electricity activity is being used
            iAe_modify = np.zeros(nAe, dtype=bool)
            for iAk in range(nAk):
                iAe_modify = iAe_modify | (
                    (np.array(activities_energy) == activities_elec[iAk]) & (energy_balance < 0)
                )

            # Modify energy prices based on flexibility and price ranges
            if shedding_guarantee[iTb] > 0:
                price_rge = np.searchsorted(price_ranges, shedding_guarantee[iTb], side='left')
                used_energy_prices[iAe_modify] = energy_prices_ranges[price_rge, iAe_modify]
            elif flexibility_form[iTb] == 'DR shifting':
                price_rge = np.searchsorted(price_ranges, flexibility_nonnegotiable[iTb], side='left')
                used_energy_prices[iAe_modify] = (
                    energy_prices[iAe_modify] * flexibility_nonnegotiable[iTb] +
                    energy_prices_ranges[price_rge, iAe_modify] * (1 - flexibility_nonnegotiable[iTb])
                )

        # Adjust investments with retrofitting
        inv_adjusted = inv_cost[iTb]
        if retrofit_potential[iTb] > 0:
            inv_adjusted = retrofit_cost[iTb]

        # Adjust investments with subsidies
        if subsidy_subject[iTb]:
            coord_subsidy_act = np.array(subsidy_activities) == activity_per_tech[iTb]
            if np.sum(coord_subsidy_act) > 0:
                subsidy_values_array = np.array(subsidy_values)
                subsidy_effect = 1 + subsidy_values_array[coord_subsidy_act, iP] / 100
                inv_adjusted *= subsidy_effect

        # Adjust voms with feed-in subsidies and taxes
        vom_adjusted = vom_cost[iTb]

        # Adjust for feed-in subsidies
        if feedin_subject[iTb]:
            coord_feedin_act = np.array(feedin_activities) == activity_per_tech[iTb]
            if np.sum(coord_feedin_act) > 0:
                feedin_values_array = np.array(feedin_values)
                feedin_effect = feedin_values_array[coord_feedin_act, iP]
                vom_adjusted -= feedin_effect

            # Adjust other activities
            activity_balance_vector = technology_balance > 0
            activity_balance_coord = np.where(activity_balance_vector)[0]
            nAB = np.sum(activity_balance_vector)
            if nAB > 0:
                for iAB in range(nAB):
                    iA_cogen = activity_balance_coord[iAB]
                    coord_feedin_act = np.array(feedin_activities) == activities_names[iA_cogen]
                    if np.sum(coord_feedin_act) > 0:
                        feedin_effect = feedin_values[coord_feedin_act, iP] * technology_balance[iA_cogen]
                        vom_adjusted -= feedin_effect

        # Adjust for taxes (emissions only)
        activity_balance_vector = (technology_balance < 0) & iAc
        activity_balance_coord = np.where(activity_balance_vector)[0]
        nAB = np.sum(activity_balance_vector)
        if nAB > 0:
            for iAB in range(nAB):
                iA_fuel = activity_balance_coord[iAB]
                coord_taxes_act = np.array(taxes_activities) == activities_names[iA_fuel]
                if np.sum(coord_taxes_act) > 0:
                    taxes_values_array = np.array(taxes_values)
                    taxes_effect = taxes_values_array[coord_taxes_act, iP] * technology_balance[iA_fuel]
                    vom_adjusted -= taxes_effect

        # Clean fuel consumption for cogeneration cases
        fuel_consumption = -energy_balance
        fuel_consumption[fuel_consumption < 0] = 0
        if iAe[iA]:
            cogen_profile = activity_balances[iTb, iAe]
            if np.sum(cogen_profile[cogen_profile > 0]) == 0:
                cogen_share = 1
            else:
                cogen_share = activity_balances[iTb, iA] / np.sum(cogen_profile[cogen_profile > 0])
        else:
            cogen_share = 1

        # Calculate the LCOP
        inv = (inv_adjusted * annuity_fact[iTb] / cap2act[iTb]).item()
        fom = (fom_cost[iTb] / cap2act[iTb]).item()
        vom = (cogen_share * vom_adjusted).item()
        fuels = (cogen_share * np.sum(fuel_consumption * used_energy_prices)).item()
        emissions = (-cogen_share * np.sum(technology_balance[iAc] * emission_prices)).item()

        # Save the LCOPs
        tech_lcops_matrix[iTb, :] = [inv, fom, vom, fuels, emissions]
        tech_lcops[iTb] = np.sum(tech_lcops_matrix[iTb, :])

    # Save the variables
    technologies['balancers']['lcops']['values'][:, iP] = tech_lcops
    # Attention: Values are correct, but sometimes the tech_lcops_matrix contains "-0". I'm not sure if this is a problem...
    technologies['balancers']['lcops']['matrix'][:, :, iP] = tech_lcops_matrix

    return technologies