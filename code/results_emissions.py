import numpy as np

def results_emissions(dimensions, types, activities, technologies, results):
    # Extract parameters
    nP = dimensions['nP']
    nA = dimensions['nA']
    sectors = types['sectors']
    activities_names = activities['names']
    emission_targets = activities['emissions']['targets']
    activity_balances = technologies['balancers']['activity_balances']
    tech_sector = technologies['balancers']['sectors']
    tech_subsector = technologies['balancers']['subsectors']
    tech_use = technologies['balancers']['use']['yearly']
    
    # Accounted emissions coordinates
    coord_act = np.zeros(nA, dtype=bool)
    for target in emission_targets:
        coord_act = np.logical_or(coord_act, np.array(activities_names) == target)
    emission_balances = -np.sum(activity_balances[:, coord_act], axis=1)
    
    # Remove emissions account (offsetting) from the total
    coord_emissions = ((np.array(tech_subsector) == 'National ETS') | (np.array(tech_subsector) == 'National nETS'))
    coord_ccs = np.array(tech_subsector) == 'CCUS Storage'
    emission_balances[coord_emissions] = 0
    
    # Obtain evolution of sectoral emissions, both positive and negative
    nS = len(sectors)
    emissions_sector_pos = np.zeros((nS, nP))
    emissions_sector_neg = np.zeros((nS, nP))
    for iS, sector in enumerate(sectors):
        # Identify sectoral technologies
        coord_tech = np.array(tech_sector) == sector
        
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
    results['emissions'] = emissions
    results['emissions_sector_pos'] = emissions_sector_pos
    results['emissions_sector_neg'] = emissions_sector_neg
    results['emissions_stored'] = emissions_stored
    
    return results
