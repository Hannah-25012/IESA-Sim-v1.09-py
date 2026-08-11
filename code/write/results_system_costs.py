# File to obtain the system costs of the energy transition
import numpy as np

def results_system_costs(dimensions, parameters, activities, technologies, results):

    # Extract parameters
    nP = dimensions['nP']
    nCc = dimensions['nCc']
    exports_value = parameters.exports_value
    activity_entities = activities.entities
    tech_entities = technologies.balancers.entities
    infra_entities = technologies.infra.entities
    energy_prices = activities.energies.prices.yearly
    inv_cost = technologies.balancers.costs.investments
    fom_cost = technologies.balancers.costs.foms
    vom_cost = technologies.balancers.costs.voms
    annuity_fact = technologies.balancers.costs.annuity
    tech_use = technologies.balancers.use.yearly
    tech_stock = technologies.balancers.stocks.evolution
    inv_cost_infra = technologies.infra.costs.investments
    fom_cost_infra = technologies.infra.costs.foms
    annuity_fact_infra = technologies.infra.costs.annuity
    tech_stock_infra = technologies.infra.stocks.evolution
    energy_exports = results.exports
    cost_categories = results.costs.categories

    # Obtain system costs for each indicator
    system_costs = np.zeros((nCc, nP))

    # Capital costs
    cost_coord = np.array(cost_categories) == 'capital'
    for iP in range(nP):

        # CHECK: Note of Manuel - % Simple definition based on stocks and current overnight costs
        # (although not the most accurate definition, it allows for a better comparisson against IESA-Opt) for the prev definition: v1.08

        # Include Balancing technologies
        for tech in tech_entities:
            # subsector='Undispatched' (VOLL, Value of Lost Load) and
            # "Indirect - <gas> <sector>" (IESA-Opt's own baseline non-
            # energy-GHG accounting) aren't physical capital at all - VOLL is
            # a scarcity-price placeholder and "Indirect -" technologies just
            # track what gets emitted before any abatement choice - see
            # invest_techstocks_def.py's own guaranteed-availability comments
            # for why their tech_stock is forced to a large, constant value
            # every period. This capital-cost formula charges an annualized
            # cost for *holding* tech_stock every period regardless of when
            # it was built, so forcing a large stock onto an accounting
            # placeholder charges a large, recurring, entirely fictitious
            # capital cost for it every single period (confirmed: one merged
            # run charged 71.7 BEUR for VOLL's stock alone in a single
            # period - not a one-time investment artifact, since zeroing the
            # separate technology_investments bookkeeping for these same
            # technologies left this number completely unchanged). Neither
            # technology exists in IESA-Sim's own native workbook with a
            # forced stock this large (VOLL's own native stock_initial is
            # already its full intended value, and "Indirect -" doesn't
            # exist there at all), so excluding them here has no native Sim
            # effect - it only removes a phantom charge unique to
            # unmatched Opt-sourced accounting placeholders.
            if tech.subsector == 'Undispatched' or tech.name.startswith('Indirect - '):
                continue
            system_costs[cost_coord, iP] += annuity_fact[tech.idx] * tech_stock[tech.idx, iP] * inv_cost[tech.idx, iP]

        # Include Infrastructure technologies
        for infra in infra_entities:
            system_costs[cost_coord, iP] += annuity_fact_infra[infra.idx] * tech_stock_infra[infra.idx, iP] * inv_cost_infra[infra.idx, iP]

    # Fixed costs
    cost_coord = np.array(cost_categories) == 'fixed operational'
    system_costs[cost_coord, :] = np.sum(tech_stock * fom_cost, axis=0)
    system_costs[cost_coord, :] += np.sum(tech_stock_infra * fom_cost_infra, axis=0)

    # Variable costs
    cost_coord = np.array(cost_categories) == 'variable'
    coord_primary = np.array([tech.category == 'Primary' for tech in tech_entities]).astype(int)
    coord_emissions = np.array([tech.category == 'Emission' for tech in tech_entities]).astype(int)
    coord_external = np.array([tech.category == 'External' for tech in tech_entities]).astype(int)
    nTb = len(tech_entities)
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
    energy_label_filtered = np.array([a.label for a in activity_entities if a.is_energy])
    coord_synfuels = (energy_label_filtered == 'Synfuels')  # CHECK: computed but not used later
    coord_oil = np.array([tech.activity is not None and tech.activity.name == 'Crude Oil' for tech in tech_entities])
    if exports_value > 0:
        energy_exports_mult = energy_exports
        system_costs[cost_coord, :] = - np.sum(energy_exports_mult, axis=0) * exports_value * vom_cost[coord_oil, :]
    else:
        system_costs[cost_coord, :] = - np.sum(energy_exports * energy_prices, axis=0)

    # Save Variables
    results.costs.system = system_costs

    return results
