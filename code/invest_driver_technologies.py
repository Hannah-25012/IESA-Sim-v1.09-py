import numpy as np

def invest_driver_technologies(dimensions, activities, technologies, 
                               tech_stock_exist, investment_potential, activity_gap, tech_choices, iP):
    
    # Extract Parameters
    nA = dimensions['nA']
    nTb = dimensions['nTb']
    activities_names = activities['names']
    activities_driver = activities['drivers']['names']
    activity_per_tech = technologies['balancers']['activities']
    cap2act = technologies['balancers']['cap2acts']
    shedding_guarantee = technologies['balancers']['shedding']['guarantee']
    tech_stock_max = technologies['balancers']['stocks']['max'][:, iP]
    tech_choices_LCOP_order = technologies['balancers']['choices_lcop_order'][:, iP]

    # Calculate the shedding adjustment
    coord_shedding_tech = shedding_guarantee > 0
    shedding_adjustment = np.ones(nTb)
    shedding_adjustment[coord_shedding_tech] = shedding_guarantee[coord_shedding_tech]

    # For every activity, add sufficient capacity
    new_investments = np.zeros((nTb,1))
    for iA in range(nA):
        if activities_names[iA] in activities_driver:
            activity_per_tech = np.array(activity_per_tech)
            tech_coord = np.char.equal(activity_per_tech, activities_names[iA])
            
            nT = np.sum(tech_coord)
            cap2act_temp = cap2act[tech_coord]
            shedding_adjustment_temp = shedding_adjustment[tech_coord]

            # Check if the activity has technologies
            if nT > 0:

                # Check investment potential constraint
                choices_investments = (activity_gap[iA] * tech_choices[tech_coord] / 
                                        cap2act_temp / shedding_adjustment_temp)
                valid_investments = np.maximum(
                    np.minimum(investment_potential[tech_coord], choices_investments), 0
                )

                # Check sufficiency
                remaining_gap = activity_gap[iA] - np.sum(valid_investments * cap2act_temp * shedding_adjustment_temp)
                fill_investments = np.zeros((nT,1))  # Preallocate
                if remaining_gap > 0:
                    tech_room = (tech_stock_max[tech_coord] - tech_stock_exist[tech_coord] - valid_investments)
                    cand_availability = tech_room > 0
                    order_search = tech_choices_LCOP_order[tech_coord]
                    iT_search = 1
                    while remaining_gap > 0:
                        iT = np.where(order_search == iT_search)[0]
                        if cand_availability[iT]:
                            fill_investments[iT] = min(
                                remaining_gap / cap2act_temp[iT] / shedding_adjustment_temp[iT],
                                tech_room[iT]
                            )
                        remaining_gap -= fill_investments[iT] * cap2act_temp[iT] * shedding_adjustment_temp[iT]
                        if remaining_gap < 1e-6:
                            remaining_gap = 0
                        iT_search += 1
                        if iT_search > nT and remaining_gap > 0:
                            tech_room_new = (tech_stock_max[tech_coord] -
                                             tech_stock_exist[tech_coord] - valid_investments - fill_investments)
                            indices = np.where(tech_room_new == np.max(tech_room_new))[0]
                            iT = indices[0] if len(indices) > 0 else None
                            fill_investments[iT, 0] += remaining_gap[0, 0] / cap2act_temp[iT] / shedding_adjustment_temp[iT]
                            # fill_investments[iT] += remaining_gap / cap2act_temp[iT] / shedding_adjustment_temp[iT]
                            remaining_gap = 0

                valid_investments = valid_investments.reshape(-1, 1)

                updated_investments = valid_investments + fill_investments
                new_investments[tech_coord, 0] = np.maximum(updated_investments.flatten(), 0)

    # Adjust investments in shedding technologies 
    # reshape to 2D array
    new_investments[coord_shedding_tech] /= shedding_guarantee[coord_shedding_tech].reshape(-1, 1)

    # Save variables
    # reshape to 2D array
    tech_stock_exist = tech_stock_exist.reshape(-1, 1)
    tech_stock_new = tech_stock_exist + new_investments

    return tech_stock_new, new_investments