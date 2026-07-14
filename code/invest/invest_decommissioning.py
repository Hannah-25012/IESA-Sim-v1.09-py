# File to calculate decomissioning in future years based on investments made in the current period
import numpy as np

def invest_decommissioning(dimensions, activities, technologies, forced_decommissionings,
                          retrofit_sources, retrofit_options, retrofit_potential, tech_stock_available, iP):
    # tech_stock_available must be net of this period's already-planned
    # decommissioning (mod2_invest passes tech_stock_new_original, not the raw
    # period-start stock) - retrofit_potential was computed against that same
    # net figure, so checking availability here against anything else would
    # let planned decommissioning and retrofit-sourced decommissioning both
    # draw against the same stock and jointly overdraw it.

    # Extract parameters
    nP = dimensions['nP']
    nTb = dimensions['nTb']
    periods = activities.periods
    tech_entities = technologies.balancers.entities
    tech_stock = technologies.balancers.stocks.evolution[:, iP]
    investments = technologies.balancers.investments[:, iP]
    decommissionings = technologies.balancers.decommissionings

    # Determine the expiracy of each technology
    for tech in tech_entities:
        expiracy = periods[iP] + tech.lifetime
        if (expiracy <= periods[-1]) and (investments[tech.idx] > 0):
            expiracy_coord = periods == min([p for p in periods if p >= expiracy])
            decommissionings[tech.idx, expiracy_coord] += investments[tech.idx]

    # Adjust the decommissionings of retrofittings
    retrofitting_decommissionings = np.zeros((nTb,1))  # Preallocate
    for tech in tech_entities:

        # Check if the technology is retrofittable and if there were investments
        if (retrofit_potential[tech.idx] > 0) and (investments[tech.idx] > 0):

            # Keep track of what has to be decommissioned
            to_decommission_still = investments[tech.idx]

            # Decommission technology one by one, in the order retrofit_sources
            # was built (cheapest sources first - see invest_retrofit_potential)
            options = retrofit_sources[tech.idx]
            iOpts = 0
            while to_decommission_still > 0:

                # Identify the technology to be decommissioned and decommission
                if iOpts >= len(options) or options[iOpts] is None:
                    break
                source_tech = options[iOpts]
                iOpts += 1

                available_stock = float(tech_stock_available[source_tech.idx])
                to_decommission_now = min(to_decommission_still, available_stock)
                to_decommission_still -= to_decommission_now

                # Save the decision
                retrofitting_decommissionings[source_tech.idx] = to_decommission_now

    # Adjust the stocks with the new period decommissionings
    tech_stock = tech_stock.reshape(-1,1)
    tech_stock -= retrofitting_decommissionings  # Note that forced decommissionings were already removed

    # Adjust the new decommissionings from the expected future decommissionings as these were prematurely carried out
    new_decommissionings = forced_decommissionings + retrofitting_decommissionings

    # Adjust the decommissioning matrix
    decommissionings[:, iP] += new_decommissionings.flatten()
    for tech in tech_entities:

        # Remove future planned decommissionings due to forwarding
        to_remove_still = float(new_decommissionings[tech.idx])
        iP_iter = iP + 1

        # Advance period by period until sufficient
        while (to_remove_still > 0) and (iP_iter < nP):

            # Check how much to remove from the period
            to_remove_now = min(decommissionings[tech.idx, iP_iter], to_remove_still)

            # Remove from the matrix
            decommissionings[tech.idx, iP_iter] -= to_remove_now

            # Advance in the while loop
            to_remove_still -= to_remove_now
            iP_iter += 1

    # Save the results
    technologies.balancers.decommissionings = decommissionings
    technologies.balancers.stocks.evolution[:, iP] = tech_stock.flatten()

    return technologies
