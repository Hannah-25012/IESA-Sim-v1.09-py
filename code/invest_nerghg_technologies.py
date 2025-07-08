import numpy as np
from invest_nerghg_sufficiency import invest_nerghg_sufficiency

def invest_nerghg_technologies(dimensions, activities, technologies, 
                                techstock_exist, investment_potential, tech_choices, iP):

    # Extract Parameters
    nA = dimensions['nA']
    nTb = dimensions['nTb']
    activities_names = activities['names']
    activities_emission = activities['emissions']['names']
    activity_per_tech = technologies['balancers']['activities']
    tech_sector = technologies['balancers']['sectors']
    cap2act = technologies['balancers']['cap2acts']
    techstock_max = technologies['balancers']['stocks']['max'][:, iP]
    tech_choices_lcop_order = technologies['balancers']['choices_lcop_order'][:, iP]

    # Initialize investments
    investments = np.zeros((nTb,1))

    # Determine the energy gap for current stock
    emission_gap = invest_nerghg_sufficiency(
        dimensions, activities, technologies, techstock_exist, True, iP
    )

    # Check for nER-GHG technologies
    techs_nerghg = np.array(tech_sector) == 'nER GHG'

    activity_per_tech = np.array(activity_per_tech)
    cap2act = np.array(cap2act)

    # Determine required investments
    for iA in range(nA):
        if activities_names[iA] in activities_emission:

            # Check if the activity has an emission gap to fill
            if emission_gap[iA] > 0:
                match_array = activity_per_tech == activities_names[iA]
                tech_coord = match_array & techs_nerghg
                nT = np.sum(tech_coord)

                # Check if there are nER-GHG technologies to invest in
                if nT > 0:
                    cap2act_temp = cap2act[tech_coord]

                    # Check max stock constraint
                    choices_investments = emission_gap[iA] * tech_choices[tech_coord] / cap2act_temp
                    valid_investments = np.maximum(
                        np.minimum(investment_potential[tech_coord], choices_investments), 0
                    )

                    # Check sufficiency
                    remaining_gap = emission_gap[iA] - np.sum(valid_investments * cap2act_temp)
                    fill_investments = np.zeros((nT,1))  # Preallocate
                    other_investments = np.zeros((nT,1))  # Preallocate

                    if remaining_gap > 0:
                        tech_room = techstock_max[tech_coord].flatten() - techstock_exist[tech_coord].flatten() - valid_investments.flatten()
                        cand_availability = tech_room > 0
                        order_search = tech_choices_lcop_order[tech_coord]
                        iT_search = 1

                        while remaining_gap > 0:
                            iT = np.where(order_search == iT_search)[0]

                            if iT.size > 0:
                                selected_index = iT[0]
                            
                                if cand_availability[selected_index]:
                                    # Compute how much can be invested
                                    remaining_gap_ratio = remaining_gap / cap2act_temp[selected_index]
                                    available_tech_room = tech_room[selected_index]
                                        
                                    fill_investments[selected_index] = min(remaining_gap_ratio, available_tech_room)
                                        
                                    # Update remaining gap
                                    remaining_gap -= fill_investments[selected_index] * cap2act_temp[selected_index]
                                
                                # In MATLAB, the loop increments iT_search each time and breaks if it surpasses nT
                                iT_search += 1
                                if iT_search > nT:
                                    break
                            
                            
                            if remaining_gap < 1e-6:
                                remaining_gap = 0
                            iT_search += 1
                            if iT_search > nT:
                                # Remaining gap will be emitted (first nER-GHG technology)
                                other_investments[0,0] = remaining_gap
                                remaining_gap = 0

                        # Declare investments
                        valid_investments = valid_investments.reshape(-1, 1)
                        fill_investments = fill_investments.reshape(-1, 1)
                        other_investments = other_investments.reshape(-1, 1)

                        investments[tech_coord] = valid_investments + fill_investments + other_investments

    # Adjust investments and new stocks
    techstock_new = techstock_exist + investments

    # To debug: Print the last 30 rows of the investments array
    # print(investments[-50:])

    return techstock_new, investments