# File to process emission charts
import numpy as np

def results_emissions(dimensions, types, activities, technologies, results):

    # Extract parameters
    nP = dimensions['nP']
    sectors = types.sectors
    activity_entities = activities.entities
    tech_entities = technologies.balancers.entities
    emission_targets = activities.emissions.targets
    activity_balances = technologies.balancers.activity_balances
    tech_use = technologies.balancers.use.yearly

    # Accounted emissions coordinates. No leading minus: activity_balances is
    # now positive-for-emitters (IESA-Opt convention), so the raw sum already
    # is "emitted amount", exactly as the negated sum used to be.
    coord_act = np.array([a.name in emission_targets for a in activity_entities])
    emission_balances = np.sum(activity_balances[:, coord_act], axis=1)

    # Remove emissions account (offsetting) from the total
    coord_emissions = np.array([tech.subsector in ('National ETS', 'National nETS') for tech in tech_entities])
    coord_ccs = np.array([tech.subsector == 'CCUS Storage' for tech in tech_entities])
    emission_balances[coord_emissions] = 0

    # Obtain evolution of sectoral emissions, both positive and negative
    nS = len(sectors)
    emissions_sector_pos = np.zeros((nS, nP))
    emissions_sector_neg = np.zeros((nS, nP))
    for iS, sector in enumerate(sectors):

        # Identify sectoral technologies
        coord_tech = np.array([tech.sector == sector for tech in tech_entities])

        # Identify positive and negative balances
        emission_balances_tech = emission_balances[coord_tech]
        emission_balances_tech_pos = emission_balances_tech.copy()
        emission_balances_tech_pos[emission_balances_tech_pos < 0] = 0
        emission_balances_tech_neg = emission_balances_tech.copy()
        emission_balances_tech_neg[emission_balances_tech_neg > 0] = 0

        # Add the volumes
        emissions_sector_pos[iS, :] = np.sum(tech_use[coord_tech, :] * emission_balances_tech_pos[:, np.newaxis], axis=0)
        emissions_sector_neg[iS, :] = np.sum(tech_use[coord_tech, :] * emission_balances_tech_neg[:, np.newaxis], axis=0)

    # Obtain total emissions
    emissions = np.sum(emissions_sector_pos, axis=0) + np.sum(emissions_sector_neg, axis=0)

    # Calculate stored emissions
    emissions_stored = np.sum(tech_use[coord_ccs, :], axis=0)

    # Save parameters
    results.emissions = emissions
    results.emissions_sector_pos = emissions_sector_pos
    results.emissions_sector_neg = emissions_sector_neg
    results.emissions_stored = emissions_stored

    return results
