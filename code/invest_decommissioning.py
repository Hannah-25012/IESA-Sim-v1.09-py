import numpy as np

def invest_decommissioning(dimensions, activities, technologies, forced_decommissionings,
                          retrofit_sources, retrofit_options, retrofit_potential, tech_stock_exist, iP):
    # Extract parameters
    nP = dimensions['nP']
    nTb = dimensions['nTb']
    periods = activities['periods']
    tech_balancers = technologies['balancers']['ids']
    ec_lifetime = technologies['balancers']['costs']['lifetimes']
    tech_stock = technologies['balancers']['stocks']['evolution'][:, iP]
    investments = technologies['balancers']['investments'][:, iP]
    decommissionings = technologies['balancers']['decommissionings']

    # Determine the expiracy of each technology
    for iTb in range(nTb):
        expiracy = periods[iP] + ec_lifetime[iTb]
        if (expiracy <= periods[-1]) and (investments[iTb] > 0):
            expiracy_coord = periods == min([p for p in periods if p >= expiracy])
            decommissionings[iTb, expiracy_coord] += investments[iTb]

    # Adjust the decommissionings of retrofittings
    retrofitting_decommissionings = np.zeros((nTb,1))  # Preallocate
    for iTb in range(nTb):

        # Check if the technology is retrofittable and if there were investments
        if (retrofit_potential[iTb] > 0) and (investments[iTb] > 0):

            # Keep track of what has to be decommissioned
            to_decommission_still = investments[iTb]

            # Decommission technology one by one
            nOpts = retrofit_options[iTb]
            iOpts = 0
            while to_decommission_still > 0:

                iOpts += 1

                # Identify the technology to be decommissioned and decommission
                source_id = retrofit_sources[iTb][iOpts - 1]

                mask = (tech_balancers == source_id)
                # Defensive check – helps discover data issues early
                if not np.any(mask):
                    break
                available_stock = float(tech_stock_exist[mask]) 
                to_decommission_now = min(to_decommission_still, available_stock)
                to_decommission_still -= to_decommission_now
                

                # Save the decision
                # retrofitting_decommissionings[tech_coord] += to_decommission_now
                retrofitting_decommissionings[mask] = to_decommission_now

    # Adjust the stocks with the new period decommissionings
    tech_stock = tech_stock.reshape(-1,1)
    tech_stock -= retrofitting_decommissionings  # Note that forced decommissionings were already removed

    # Adjust the new decommissionings from the expected future decommissionings as these were prematurely carried out
    new_decommissionings = forced_decommissionings + retrofitting_decommissionings

    # Adjust the decommissioning matrix
    decommissionings[:, iP] += new_decommissionings.flatten()
    for iTb in range(nTb):
        # debugging:
        # if iTb == 398:
        #     print(f"Reached iTb = {iTb}. Press Enter to continue.")
        #     input()  

        # Remove future planned decommissionings due to forwarding
        # to_remove_still = new_decommissionings[iTb]
        to_remove_still = float(new_decommissionings[iTb])  # Copy value to avoid modifying original
        iP_iter = iP + 1

        # Advance period by period until sufficient
        while (to_remove_still > 0) and (iP_iter < nP):

            # Check how much to remove from the period
            to_remove_now = min(decommissionings[iTb, iP_iter], to_remove_still)

            # Remove from the matrix
            decommissionings[iTb, iP_iter] -= to_remove_now

            # Advance in the while loop
            to_remove_still -= to_remove_now
            iP_iter += 1

    # Save the results
    technologies['balancers']['decommissionings'] = decommissionings
    technologies['balancers']['stocks']['evolution'][:, iP] = tech_stock.flatten()

    return technologies