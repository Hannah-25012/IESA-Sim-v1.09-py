import numpy as np
from invest_ccus_sufficiency import invest_ccus_sufficiency

def invest_ccus_technologies(dimensions, activities, technologies, 
                                techstock_exist, investment_potential, tech_choices, iP):
    
    # Extract Parameters
    nA = dimensions['nA']
    nTb = dimensions['nTb']
    activities_names = activities['names']
    activities_emission = activities['emissions']['names']
    activity_per_tech = technologies['balancers']['activities']
    tech_subsector = technologies['balancers']['subsectors']
    cap2act = technologies['balancers']['cap2acts']
    techstock_max = technologies['balancers']['stocks']['max'][:, iP]

    # Initialize investments
    investments = np.zeros((nTb,1))

    # Determine the energy gap for current stock
    ccus_gap = invest_ccus_sufficiency(
        dimensions, activities, technologies, techstock_exist, False, iP
    )

    # Check for CCUS technologies
    techs_ccus = np.array(tech_subsector) == 'CCUS'

    activity_per_tech = np.array(activity_per_tech)
    cap2act = np.array(cap2act)

    # Determine required investments
    for iA in range(nA):
        if activities_names[iA] in activities_emission:

            # Check if the activity has an emission gap to fill
            if ccus_gap[iA] > 0:
                match_array = activity_per_tech == activities_names[iA]
                tech_coord = match_array & techs_ccus
                nT = np.sum(tech_coord)

                # Check if there are nER-GHG technologies to invest in
                if nT > 0:
                    cap2act_temp = cap2act[tech_coord]

                    # Check max stock constraint
                    choices_investments = ccus_gap[iA] * tech_choices[tech_coord] / cap2act_temp
                    valid_investments = np.maximum(
                        np.minimum(investment_potential[tech_coord], choices_investments), 0
                    )

                    # Check sufficiency
                    remaining_gap = ccus_gap[iA] - np.sum(valid_investments * cap2act_temp)
                    fill_investments = np.zeros((nT,1))  # Preallocate
                    other_investments = np.zeros((nT,1))  # Preallocate

                    if remaining_gap > 0:
                        tech_room = techstock_max[tech_coord].flatten() - techstock_exist[tech_coord].flatten() - valid_investments.flatten()
                        # change line!
                        cand_availability = (tech_choices[tech_coord] > 0) * (tech_room > 0)
                        
                        iT = 1

                        while remaining_gap > 0:
                        
                            if len(iT) > 0 and cand_availability[iT].any():
                                selected_index = iT[0]
                                selected_index = int(selected_index)
                                remaining_gap_ratio = remaining_gap / cap2act_temp[selected_index].item()
                                available_tech_room = tech_room[selected_index].item()
                                fill_investments[selected_index] = min(remaining_gap_ratio, available_tech_room)
                                
                            remaining_gap -= fill_investments[selected_index] * cap2act_temp[selected_index]
                            if remaining_gap < 1e-6:
                                remaining_gap = 0
                            iT += 1
                            if iT > nT:
                                # Display a warning message
                                print(f"!!!!Warning: There is not enough available max capacity to satisfy the demand for {activities[iA]}, the remaining gap is {-remaining_gap:6.2f} UoA")
                                remaining_gap = 0

                        # Declare investments
                        valid_investments = valid_investments.reshape(-1, 1)
                        fill_investments = fill_investments.reshape(-1, 1)
                        other_investments = other_investments.reshape(-1, 1)

                        investments[tech_coord] = valid_investments + fill_investments + other_investments

    # Adjust investments and new stocks
    techstock_new = techstock_exist + investments

    # To debug: Print the last 30 rows of the investments array
    # print(investments[-30:])

    return techstock_new, investments