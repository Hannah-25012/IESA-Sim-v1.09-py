import numpy as np

def results_system_costs(dimensions, parameters, activities, technologies, results):
    # Extract parameters
    nP = dimensions['nP']
    nTb = dimensions['nTb']
    nTi = dimensions['nTi']
    nCc = dimensions['nCc']
    exports_value = parameters['exports_value']
    activity_label = activities['labels']
    activity_type_act = activities['types']
    energy_prices = activities['energies']['prices']['yearly']
    activity_per_tech = technologies['balancers']['activities']
    tech_categories = technologies['balancers']['categories']
    inv_cost = technologies['balancers']['costs']['investments']
    fom_cost = technologies['balancers']['costs']['foms']
    vom_cost = technologies['balancers']['costs']['voms']
    annuity_fact = technologies['balancers']['costs']['annuity']
    tech_use = technologies['balancers']['use']['yearly']
    tech_stock = technologies['balancers']['stocks']['evolution']
    inv_cost_infra = technologies['infra']['costs']['investments']
    fom_cost_infra = technologies['infra']['costs']['foms']
    annuity_fact_infra = technologies['infra']['costs']['annuity']
    tech_stock_infra = technologies['infra']['stocks']['evolution']
    energy_exports = results['exports']
    cost_categories = results['costs']['categories']
    
    # Obtain system costs for each indicator
    system_costs = np.zeros((nCc, nP))
    
    # Capital costs
    cost_coord = np.array(cost_categories) == 'capital'
    for iP in range(nP):
        # Include Balancing technologies
        for iTb in range(nTb):
            system_costs[cost_coord, iP] += annuity_fact[iTb] * tech_stock[iTb, iP] * inv_cost[iTb, iP]
        # Include Infrastructure technologies
        for iTi in range(nTi):
            system_costs[cost_coord, iP] += annuity_fact_infra[iTi] * tech_stock_infra[iTi, iP] * inv_cost_infra[iTi, iP]
    
    # Fixed costs
    cost_coord = np.array(cost_categories) == 'fixed operational'
    system_costs[cost_coord, :] = np.sum(tech_stock * fom_cost, axis=0)
    system_costs[cost_coord, :] += np.sum(tech_stock_infra * fom_cost_infra, axis=0)
    
    # Variable costs
    cost_coord = np.array(cost_categories) == 'variable'
    coord_primary = (np.array(tech_categories) == 'Primary').astype(int)
    coord_emissions = (np.array(tech_categories) == 'Emission').astype(int)
    coord_external = (np.array(tech_categories) == 'External').astype(int)
    coord_vom = np.ones((nTb, 1)) - coord_primary.reshape(nTb, 1) - coord_emissions.reshape(nTb, 1) - coord_external.reshape(nTb, 1)
    mask_vom = (coord_vom * np.ones((1, nP))) > 0
    system_costs[cost_coord, :] = np.sum(tech_use * vom_cost * mask_vom, axis=0)
    
    # Fuel costs
    cost_coord = np.array(cost_categories) == 'fuels'
    coord_fuel = coord_primary
    mask_fuel = (coord_fuel.reshape(nTb, 1) * np.ones((1, nP))) > 0
    system_costs[cost_coord, :] = np.sum(tech_use * vom_cost * mask_fuel, axis=0)
    
    # Emission costs
    cost_coord = np.array(cost_categories) == 'emissions'
    mask_emissions = (coord_emissions.reshape(nTb, 1) * np.ones((1, nP))) > 0
    system_costs[cost_coord, :] = np.sum(tech_use * vom_cost * mask_emissions, axis=0)
    
    # Export revenues
    cost_coord = np.array(cost_categories) == 'exporting revenues'
    mask_activity = (np.array(activity_type_act) == 'Energy') | (np.array(activity_type_act) == 'Fix Energy')
    energy_label_filtered = np.array(activity_label)[mask_activity]
    coord_synfuels = (energy_label_filtered == 'Synfuels')
    coord_oil = (np.array(activity_per_tech) == 'Crude Oil')
    if exports_value > 0:
        energy_exports_mult = energy_exports
        system_costs[cost_coord, :] = - np.sum(energy_exports_mult, axis=0) * exports_value * vom_cost[coord_oil, :]
    else:
        system_costs[cost_coord, :] = - np.sum(energy_exports * energy_prices, axis=0)
    
    # Save Variables
    results['costs']['system'] = system_costs
    
    return results
