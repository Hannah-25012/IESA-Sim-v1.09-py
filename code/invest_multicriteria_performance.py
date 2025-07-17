import numpy as np

def invest_multicriteria_performance(dimensions, activities, technologies, agents, iP):
    # Extract parameters
    nA = dimensions['nA']
    nTb = dimensions['nTb']
    nMC = dimensions['nMC']
    activities_names = activities['names']
    activity_types = activities['types']
    activity_per_tech = technologies['balancers']['activities']
    activity_balances = technologies['balancers']['activity_balances']
    buffer_capacity = technologies['balancers']['buffers']['capacity']
    tech_LCOPs = technologies['balancers']['lcops']['values'][:, iP]
    social_perception_tech = technologies['balancers']['agents']['social_perception']
    perceived_complexity_tech = technologies['balancers']['agents']['complexity']
    multicriteria_performance_tech = technologies['balancers']['mca']['matrix'][:, :, iP]
    multicriteria_categories = agents['criteria']['categories']

    # Identify multi-criteria coordinates
    iMC1 = multicriteria_categories.index('Social Attitude')
    iMC2 = multicriteria_categories.index('Emissions performance')
    iMC3 = multicriteria_categories.index('Cost performance')
    iMC4 = multicriteria_categories.index('Complexity')

    # Identify emissions coordinates
    iAc = [i for i, t in enumerate(activity_types) if t == 'Emission']

    # Preallocate emissions
    emissions = np.zeros(nTb)

    # Calculate indexes for each parameter
    for iTb in range(nTb):
        # Extract the balance of the technology
        technology_balance = np.copy(activity_balances[iTb, :])
        technology_balance[activities_names.index(activity_per_tech[iTb])] = 0

        # Calculate emissions
        emissions[iTb] = -np.sum(technology_balance[iAc])

        # Retrieve social perception parameter
        if social_perception_tech[iTb] == 'Negative':
            multicriteria_performance_tech[iTb, iMC1] = 0
        elif social_perception_tech[iTb] == 'Neutral':
            multicriteria_performance_tech[iTb, iMC1] = 0.5
        elif social_perception_tech[iTb] == 'Positive':
            multicriteria_performance_tech[iTb, iMC1] = 1

        # Retrieve perceived complexity parameter
        if perceived_complexity_tech[iTb] == 'Low':
            multicriteria_performance_tech[iTb, iMC4] = 1
        elif perceived_complexity_tech[iTb] == 'Med':
            multicriteria_performance_tech[iTb, iMC4] = 0.5
        elif perceived_complexity_tech[iTb] == 'High':
            multicriteria_performance_tech[iTb, iMC4] = 0

    # Identify emission and LCOP ranges for all technologies with the same main activity
    for iA in range(nA):
        # Identify competing technologies
        icoord = [i for i, act in enumerate(activity_per_tech) if act == activities_names[iA]]

        # Check if the technology is a buffer
        no_buffer_coord = [buffer_capacity[i] == 0 for i in icoord]
        icoord = [i for i, valid in zip(icoord, no_buffer_coord) if valid]

        if len(icoord) > 0:
            # Identify vectors of emissions and LCOPs
            emissions_vec = emissions[icoord]
            LCOP_vec = tech_LCOPs[icoord]

            # Identify maximum and minimum values per technology
            LCOP_min = np.min(LCOP_vec)
            LCOP_max = LCOP_min + 0.5 * abs(LCOP_min) + 1e-6
            emissions_min = np.min(emissions_vec)
            # Attention: Matlab takes only the first element of emissions_vec (not the maximum of the vector) and then compares to 0. 
            # To match this (for comparison), we need to remove np.max
            # emissions_max = max(np.max(emissions_vec), 0) + 1e-6
            emissions_max = max(emissions_vec[0], 0) + 1e-6

            # Calculate the decreasing linear functions from 1 to 0
            # For emissions
            multicriteria_performance_tech[icoord, iMC2] = 1 - (
                (emissions_vec - emissions_min) / (emissions_max - emissions_min)
            )

            # For LCOPs
            multicriteria_performance_tech[icoord, iMC3] = 1 - (
                (LCOP_vec - LCOP_min) / (LCOP_max - LCOP_min)
            )

    # Remove negative results
    multicriteria_performance_tech[:, [iMC2, iMC3]] = np.maximum(
        multicriteria_performance_tech[:, [iMC2, iMC3]], 0
    )

    # Save the multi-criteria analysis matrix
    technologies['balancers']['mca']['matrix'][:, :, iP] = multicriteria_performance_tech

    return technologies
