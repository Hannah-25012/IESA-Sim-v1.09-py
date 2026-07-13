# File to determine the multi-criteria performance of technologies
import numpy as np

def invest_multicriteria_performance(dimensions, activities, technologies, agents, iP):

    # Extract parameters
    nTb = dimensions['nTb']
    tech_entities = technologies.balancers.entities
    emission_idx = [a.idx for a in activities.entities if a.is_emission]
    tech_LCOPs = technologies.balancers.lcops.values[:, iP]
    multicriteria_performance_tech = technologies.balancers.mca.matrix[:, :, iP]
    multicriteria_categories = agents.criteria.categories

    # Identify multi-criteria coordinates
    iMC1 = multicriteria_categories.index('Social Attitude')
    iMC2 = multicriteria_categories.index('Emissions performance')
    iMC3 = multicriteria_categories.index('Cost performance')
    iMC4 = multicriteria_categories.index('Complexity')

    # Do a loop for all technologies to calculate their multi-criteria performance
    emissions = np.zeros(nTb) # Preallocate emissions
    for tech in tech_entities:

        # Extract the balance of the technology, zeroing out its own main activity
        technology_balance = np.copy(tech.activity_balances)
        if tech.activity is not None:
            technology_balance[tech.activity.idx] = 0

        # Calculate emissions
        emissions[tech.idx] = -np.sum(technology_balance[emission_idx])

        # Retrieve social perception parameter
        if tech.social_perception == 'Negative':
            multicriteria_performance_tech[tech.idx, iMC1] = 0
        elif tech.social_perception == 'Neutral':
            multicriteria_performance_tech[tech.idx, iMC1] = 0.5
        elif tech.social_perception == 'Positive':
            multicriteria_performance_tech[tech.idx, iMC1] = 1

        # Retrieve perceived complexity parameter
        if tech.complexity == 'Low':
            multicriteria_performance_tech[tech.idx, iMC4] = 1
        elif tech.complexity == 'Med':
            multicriteria_performance_tech[tech.idx, iMC4] = 0.5
        elif tech.complexity == 'High':
            multicriteria_performance_tech[tech.idx, iMC4] = 0

    # Identify emission and LCOP ranges for all technologies with the same main activity
    for activity in activities.entities:

        # Identify competing (non-buffer) technologies for this activity
        competitors = [tech for tech in activity.technologies if not tech.is_buffer]

        if len(competitors) > 0:
            icoord = [tech.idx for tech in competitors]

            # Identify vectors of emissions and LCOPs
            emissions_vec = emissions[icoord]
            LCOP_vec = tech_LCOPs[icoord]

            # Identify maximum and minimum values per technology
            LCOP_min = np.min(LCOP_vec)
            LCOP_max = LCOP_min + 0.5 * abs(LCOP_min) + 1e-6
            emissions_min = np.min(emissions_vec)
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

    # Save the multi-criteria performance matrix
    technologies.balancers.mca.matrix[:, :, iP] = multicriteria_performance_tech

    return technologies
