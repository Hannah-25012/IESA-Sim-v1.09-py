# Attention: This module is only called from the second period onwards, so still needs to be debugged.

import numpy as np

def invest_power_technologies(dimensions, parameters, activities, technologies, agents, tech_stock_exist, iP):

    # Extract Parameters
    nTb = dimensions['nTb']
    nAT = dimensions['nAT']
    powinv_SPBT_benchmark = parameters['powinv']['SPBT_benchmark']
    powinv_SPBT_min = parameters['powinv']['SPBT_min']
    powinv_CR_threshold = parameters['powinv']['CR_threshold']
    powinv_CR_min = parameters['powinv']['CR_min']
    powinv_NUF_threshold = parameters['powinv']['NUF_threshold']
    powinv_NUF_min = parameters['powinv']['NUF_min']
    activities_names = activities['names']
    activity_label = activities['labels']
    activity_agent = activities['agents']
    activity_per_tech = technologies['balancers']['activities']
    inv_cost = technologies['balancers']['costs']['investments'][:, iP]
    flexibility_form = technologies['balancers']['flexibility']['form']
    tech_stock_deploy = technologies['balancers']['stocks']['deploy']
    tech_stock = technologies['balancers']['stocks']['evolution'][:, iP]
    tech_stock_min = technologies['balancers']['stocks']['min'][:, iP]
    tech_stock_max = technologies['balancers']['stocks']['max'][:, iP]
    generator_capt_rate = technologies['balancers']['generators']['CR'][:, iP-1]
    generator_norm_ut_fact = technologies['balancers']['generators']['NUF'][:, iP-1]
    generator_cash_flow = technologies['balancers']['generators']['CF'][:, iP-1]
    multi_criteria_performance_tech = technologies['balancers']['mca']['matrix'][:, :, iP]
    agent_profiles = agents['profiles']
    multi_criteria_categories = agents['criteria']['categories']
    weights_multi_criteria = agents['criteria']['weights']
    agents_populations = agents['populations']

    # Identify the choice per activity and agent type
    iMC3 = np.where(np.array(multi_criteria_categories) == 'Cost performance')[0][0]

    # Preallocate new investments
    new_investments = np.zeros((nTb,1))

    # Loop through all technologies
    for iTb in range(nTb):

        # Identify if the main activity of the technology is electricity generation
        act_coord = np.array(activities_names) == activity_per_tech[iTb]

        if np.sum(np.array(activity_label)[act_coord] == 'Electricity') == 1 and inv_cost[iTb] > 0:

            # Check conditions for capture rates and capacity factors
            condition_1 = generator_capt_rate[iTb] >= powinv_CR_threshold and generator_norm_ut_fact[iTb] >= powinv_NUF_min
            condition_2 = generator_norm_ut_fact[iTb] >= powinv_NUF_threshold and generator_capt_rate[iTb] >= powinv_CR_min

            # Batteries are always eligible if cost-effective
            if flexibility_form[iTb] == 'Storage':
                condition_1 = True

            # Execute the condition
            if condition_1 or condition_2:

                # Calculate the SPBT of the technology
                powinv_SPBT_iTb = inv_cost[iTb] * tech_stock[iTb] / generator_cash_flow[iTb]

                # Check if the technology meets the investors' appetite
                if powinv_SPBT_iTb <= powinv_SPBT_min:

                    # Determine potential volume ranges for investments
                    range_min = 1  # GW
                    range_max = max(1, tech_stock_deploy[iTb])
                    potential_range = range_max - range_min

                    # Recalculate economic performance based on SPBT
                    multi_criteria_generator = multi_criteria_performance_tech[iTb, :].copy()
                    powinv_spbt = 1 - (powinv_SPBT_iTb - powinv_SPBT_benchmark) / (powinv_SPBT_min - powinv_SPBT_benchmark)
                    multi_criteria_generator[iMC3] = max(min(1, powinv_spbt), 0)

                    # Use the multi_criteria_performance matrix to get an indicator for each technology and actor
                    coord_agent_profile = np.array(activity_agent)[act_coord] == np.array(agent_profiles)
                    population_vector = agents_populations[coord_agent_profile, :].flatten()
                    population_vector = population_vector / np.sum(population_vector)

                    # Quantify the interest per agent and add to the investment counter
                    investment_itb = range_min
                    for iAT in range(nAT):
                        criteria_iat = np.sum(weights_multi_criteria[:, iAT] * multi_criteria_generator) / np.sum(weights_multi_criteria[:, iAT])
                        investment_itb += potential_range * criteria_iat * population_vector[iAT]

                    new_investments[iTb] = investment_itb

            # Ensure volumes satisfy tech_stock min and max constraints
            up_room = tech_stock_max[iTb] - tech_stock_exist[iTb]
            min_room = max(0, (tech_stock_min[iTb] - tech_stock_exist[iTb]))
            new_investments[iTb] = max(min(up_room, new_investments[iTb]), min_room)

    return new_investments
