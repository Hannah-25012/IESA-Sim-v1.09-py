import numpy as np
from invest_energy_sufficiency import invest_energy_sufficiency



def invest_energy_technologies(dimensions, activities, technologies, techstock_exist, 
                                investment_potential, tech_choices, iP):
    # Extract Parameters
    nI = 3  # Number of investment iterations
    nA = dimensions['nA']
    nAe = dimensions['nAe']
    nTb = dimensions['nTb']
    activities_names = activities['names']
    activities_energy = activities['energies']['names']
    activity_label = activities['labels']
    energy_scarcity = activities['energies']['scarcity'][:, iP]
    activity_per_tech = technologies['balancers']['activities']
    cap2act = technologies['balancers']['cap2acts']
    shedding_guarantee = technologies['balancers']['shedding']['guarantee']
    techstock_max = technologies['balancers']['stocks']['max'][:, iP]
    tech_choices_lcop_order = technologies['balancers']['choices_lcop_order'][:, iP]

    # Calculate the shedding adjustment
    coord_shedding_tech = shedding_guarantee > 0
    shedding_adjustment = np.ones(nTb)
    shedding_adjustment[coord_shedding_tech] = shedding_guarantee[coord_shedding_tech]

    # For every activity add sufficient capacity
    investments = np.zeros((nTb,1))
    techstock_exist_check = techstock_exist.copy()

    # debugging: apply rounding to 14 decimals (same is done in matlab)
    precision = 14

    for iI in range(nI):
        # Determine the energy gap for current stock
        # in the first investment iteration, energy_gap is correct. From the second iteration, there are some differences. Some variable is probably not updated correctly, have to find out why and how.
        energy_gap = invest_energy_sufficiency(dimensions, activities, technologies, 
                                               techstock_exist_check, False, iP)
        
        energy_gap = np.round(energy_gap, decimals=precision)

        # Determined required investments in the iteration
        iter_investments = np.zeros((nTb,1))
        for iA in range(nA):
            if activities_names[iA] in activities_energy:
                if activity_label[iA] != 'Electricity' and energy_gap[iA] > 0:
                    tech_coord = [act == activities_names[iA] for act in activity_per_tech]
                    nT = sum(tech_coord)
                    cap2act_temp = cap2act[tech_coord]
                    shedding_adjustment_temp = shedding_adjustment[tech_coord]

                    if nT > 0:
                        # Check max stock constraint
                        choices_investments = (energy_gap[iA] * tech_choices[tech_coord] / 
                                               cap2act_temp / shedding_adjustment_temp)
                        
                        choices_investments = np.round(choices_investments, decimals=precision)

                        valid_investments = np.maximum(
                            np.minimum(investment_potential[tech_coord], choices_investments), 0
                        )

                        valid_investments = np.round(valid_investments, decimals=precision)

                        # Check sufficiency
                        remaining_gap = energy_gap[iA] - \
                                        np.sum(valid_investments * cap2act_temp * shedding_adjustment_temp)
                        
                        remaining_gap = np.round(remaining_gap, decimals=precision)

                        fill_investments = np.zeros((nT,1))
                        other_investments = np.zeros((nT,1))

                        if remaining_gap > 0:
                            tech_room = (techstock_max[tech_coord].flatten() - techstock_exist[tech_coord].flatten() - 
                                         valid_investments.flatten())
                            
                            tech_room = np.round(tech_room, decimals=precision)

                            cand_availability = tech_room > 0
                            order_search = tech_choices_lcop_order[tech_coord]
                            iT_search = 1
                            # iT_search = np.array([1])

                            while remaining_gap > 0:
                                iT = np.where(order_search == iT_search)[0]
                                if len(iT) > 0 and cand_availability[iT].any():
                                    selected_index = iT[0]
                                    selected_index = int(selected_index)
                                    remaining_gap_ratio = remaining_gap / (cap2act_temp[selected_index].item() * shedding_adjustment_temp[selected_index].item())

                                    # remaining_gap_ratio = np.round(remaining_gap_ratio, decimals=precision)

                                    available_tech_room = tech_room[selected_index].item()
                                    fill_investments[selected_index] = min(remaining_gap_ratio, available_tech_room)

                                remaining_gap -= (fill_investments[iT[0]] * cap2act_temp[iT[0]] * 
                                                  shedding_adjustment_temp[iT[0]])
                                
                                remaining_gap = np.round(remaining_gap, decimals=precision)

                                if remaining_gap < 1e-6:
                                    remaining_gap = 0
                                iT_search += 1
                                
                                if iT_search > nT and remaining_gap > 0:
                                    tech_room_others = (techstock_max[tech_coord] - techstock_exist[tech_coord] -
                                                        valid_investments - fill_investments)
                                    if np.sum(tech_room_others) > 0:
                                        for iT in range(nT):
                                            other_investments[iT] = max(
                                                min(tech_room_others[iT] * cap2act_temp[iT], remaining_gap), 0
                                            ) / cap2act_temp[iT] / shedding_adjustment_temp[iT]
                                            remaining_gap -= (other_investments[iT] * cap2act_temp[iT] * 
                                                              shedding_adjustment_temp[iT])
                                    if remaining_gap > 0:
                                        if iI == nI - 1:
                                            print(f"!!!!Warning: Not enough capacity to satisfy the demand for {activities_names[iA]}. Remaining gap: {remaining_gap.item():.2f} UoA")
                                        remaining_gap = 0

                        
                        # valid_investments = valid_investments.reshape(-1, 1)
                        # fill_investments = fill_investments.reshape(-1, 1)
                        # other_investments = other_investments.reshape(-1, 1)

                        valid_investments = np.round(valid_investments.reshape(-1, 1), decimals=precision)
                        fill_investments = np.round(fill_investments.reshape(-1, 1), decimals=precision)
                        other_investments = np.round(other_investments.reshape(-1, 1), decimals=precision)

                        updated_investments = valid_investments + fill_investments + other_investments
                        # iter_investments[tech_coord,0] = np.maximum(updated_investments.flatten(), 0)
                        iter_investments[tech_coord, 0] = np.maximum(np.round(updated_investments.flatten(), decimals=precision), 0)

        # To debug: Print the last 30 rows of the investments array
        # print(iter_investments[-50:])

        techstock_exist_check += iter_investments

        techstock_exist_check = np.round(techstock_exist_check, decimals=precision)

        investments += iter_investments

        investments = np.round(investments, decimals=precision)

    techstock_new = techstock_exist + investments

    techstock_new = np.round(techstock_new, decimals=precision)

    energy_gap = invest_energy_sufficiency(dimensions, activities, technologies, techstock_new, False, iP)
    energy_gap = np.round(energy_gap, decimals=precision)

    
    energy_scarcity_bin = np.zeros((nA, 1), dtype=bool)
    coord_gap = (energy_gap > 0).reshape(-1, 1)
    coord_nonE = (np.array(activity_label) != 'Electricity').astype(np.float64).reshape(-1, 1)
    logical_indices = np.logical_and(coord_gap, coord_nonE.astype(bool))
    energy_scarcity_bin[logical_indices] = True

    tech_preference = tech_choices.copy()
    for iA in range(nA):
        if energy_scarcity_bin[iA]:
            coord_tech = [act == activities_names[iA] for act in activity_per_tech]
            coord_options = np.where(coord_tech)[0]

            options_preferences = tech_preference[coord_tech]
            max_value = np.max(options_preferences)
            options_sel = np.where(options_preferences == max_value)[0][0]

            choice_coord = coord_options[options_sel]

            investments[choice_coord] += energy_gap[iA] / cap2act[choice_coord]
            techstock_new[choice_coord] += energy_gap[iA] / cap2act[choice_coord]
            techstock_max[choice_coord] = techstock_new[choice_coord]

    energy_scarcity_add = np.zeros((nAe,1))
    for iAe in range(nAe):
        coord_activity = [act == activities_energy[iAe] for act in activities_names]
        if energy_scarcity_bin[coord_activity]:
            energy_scarcity_add[iAe] = energy_gap[coord_activity][0]


    activities['energies']['scarcity'][:, iP] = energy_scarcity + energy_scarcity_add.flatten()
    technologies['balancers']['stocks']['max'][:, iP] = techstock_max

    

    return technologies, activities, techstock_new, investments