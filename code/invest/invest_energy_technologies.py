# File to invest in energy technologies
import numpy as np
from invest_energy_sufficiency import invest_energy_sufficiency

def invest_energy_technologies(dimensions, activities, technologies, techstock_exist,
                                investment_potential, tech_choices, iP):
    # Extract Parameters
    nI = 3  # Number of investment iterations
    nA = dimensions['nA']
    nAe = dimensions['nAe']
    nTb = dimensions['nTb']
    energy_activities = [a for a in activities.entities if a.is_energy]
    cap2act = technologies.balancers.cap2acts
    shedding_guarantee = technologies.balancers.shedding.guarantee
    techstock_max = technologies.balancers.stocks.max[:, iP]
    tech_choices_lcop_order = technologies.balancers.choices_lcop_order[:, iP]

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
        energy_gap = invest_energy_sufficiency(dimensions, activities, technologies,
                                               techstock_exist_check, False, iP)
        energy_gap = np.round(energy_gap, decimals=precision)

        # Determined required investments in the iteration
        iter_investments = np.zeros((nTb,1))
        for activity in energy_activities:
            if not activity.is_electricity and energy_gap[activity.idx] > 0:

                tech_coord = np.array([t.idx for t in activity.technologies], dtype=int)
                nT = len(tech_coord)
                cap2act_temp = cap2act[tech_coord]
                shedding_adjustment_temp = shedding_adjustment[tech_coord]

                # Check if the activity has technologies to invest in
                if nT > 0:
                    # Check maximum stock constraint
                    choices_investments = (energy_gap[activity.idx] * tech_choices[tech_coord] /
                                           cap2act_temp / shedding_adjustment_temp)
                    choices_investments = np.round(choices_investments, decimals=precision)
                    valid_investments = np.maximum(
                        np.minimum(investment_potential[tech_coord], choices_investments), 0
                    )
                    valid_investments = np.round(valid_investments, decimals=precision)

                    # Check sufficiency
                    remaining_gap = energy_gap[activity.idx] - \
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

                        while remaining_gap > 0:
                            iT = np.where(order_search == iT_search)[0]
                            if len(iT) > 0 and cand_availability[iT].any():
                                selected_index = iT[0]
                                selected_index = int(selected_index)
                                remaining_gap_ratio = remaining_gap / (cap2act_temp[selected_index].item() * shedding_adjustment_temp[selected_index].item())
                                available_tech_room = tech_room[selected_index].item()
                                fill_investments[selected_index] = min(remaining_gap_ratio, available_tech_room)

                            remaining_gap -= (fill_investments[iT[0]] * cap2act_temp[iT[0]] *
                                              shedding_adjustment_temp[iT[0]])
                            remaining_gap = np.round(remaining_gap, decimals=precision)

                            if remaining_gap < 1e-6:
                                remaining_gap = 0
                            iT_search += 1

                            if iT_search > nT and remaining_gap > 0:

                                # Check for other technologies that were not chosen by agents
                                tech_room_others = (techstock_max[tech_coord] - techstock_exist[tech_coord].ravel() -
                                                    valid_investments.ravel() - fill_investments.ravel())
                                if np.sum(tech_room_others) > 0:
                                    for iT in range(nT):
                                        other_investments[iT] = max(
                                            min(tech_room_others[iT] * cap2act_temp[iT], remaining_gap), 0
                                        ) / cap2act_temp[iT] / shedding_adjustment_temp[iT]
                                        remaining_gap -= (other_investments[iT] * cap2act_temp[iT] *
                                                          shedding_adjustment_temp[iT])

                                # If still gap remains, display warning
                                if remaining_gap > 0:
                                    if iI == nI - 1:
                                        print(f"!!!!Warning: Not enough capacity to satisfy the demand for {activity.name}. Remaining gap: {remaining_gap.item():.2f} UoA")
                                    remaining_gap = 0

                    valid_investments = np.round(valid_investments.reshape(-1, 1), decimals=precision)
                    fill_investments = np.round(fill_investments.reshape(-1, 1), decimals=precision)
                    other_investments = np.round(other_investments.reshape(-1, 1), decimals=precision)
                    updated_investments = valid_investments + fill_investments + other_investments
                    iter_investments[tech_coord, 0] = np.maximum(np.round(updated_investments.flatten(), decimals=precision), 0)

        # Updated existing stock to determine gaps
        techstock_exist_check += iter_investments
        techstock_exist_check = np.round(techstock_exist_check, decimals=precision)
        investments += iter_investments
        investments = np.round(investments, decimals=precision)

    # Define investments
    techstock_new = techstock_exist + investments
    techstock_new = np.round(techstock_new, decimals=precision)

    # Check the gaps and report them
    energy_gap = invest_energy_sufficiency(dimensions, activities, technologies, techstock_new, False, iP)
    energy_gap = np.round(energy_gap, decimals=precision)

    # Check which energy forms present scarcity
    energy_scarcity_bin = np.zeros((nA, 1), dtype=bool)
    coord_gap = (energy_gap > 1e-12).reshape(-1, 1)
    is_electricity = np.array([a.is_electricity for a in activities.entities]).reshape(-1, 1)
    coord_nonE = (~is_electricity)
    logical_indices = np.logical_and(coord_gap, coord_nonE)
    energy_scarcity_bin[logical_indices] = True

    # Facilitate the resources for this energy to be satisfied
    tech_preference = tech_choices.copy()
    for activity in activities.entities:
        if energy_scarcity_bin[activity.idx]:

            # Identify possible technologies
            tech_coord = np.array([t.idx for t in activity.technologies], dtype=int)

            # An activity can have an energy gap but no technology at all to fill it
            # (e.g. an IESA-Sim-side activity name with no matching IESA-Opt-side
            # technology after a merged-database load - see dbcompare-backend's
            # /unify - such as "Natural Gas"/"Hydrogen" vs. the technology-bearing
            # "Natural Gas HD"/"Natural Gas LD" naming) - skip it like the analogous
            # nT == 0 case in invest_tech_choices_per_act.py, instead of crashing on
            # np.max of an empty array.
            if tech_coord.size == 0:
                print(f"--****There is no technology to cover the energy gap for activity: {activity.name}")
                continue

            # Preferences of thoses technologies
            options_preferences = tech_preference[tech_coord]
            max_value = np.max(options_preferences)
            options_sel = np.where(options_preferences == max_value)[0][0]

            # Identify which technology was chosen
            choice_coord = tech_coord[options_sel]

            # Fill investments and stocks
            investments[choice_coord] += energy_gap[activity.idx] / cap2act[choice_coord]
            techstock_new[choice_coord] += energy_gap[activity.idx] / cap2act[choice_coord]
            techstock_max[choice_coord] = techstock_new[choice_coord]

    # Calculate energy scarcity
    energy_scarcity_add = np.zeros((nAe,1))
    for iAe, activity in enumerate(energy_activities):
        if energy_scarcity_bin[activity.idx]:
            energy_scarcity_add[iAe] = energy_gap[activity.idx][0]

    # Save variables
    energy_scarcity = activities.energies.scarcity[:, iP].copy()
    activities.energies.scarcity[:, iP] = energy_scarcity + energy_scarcity_add.flatten()
    technologies.balancers.stocks.max[:, iP] = techstock_max

    return technologies, activities, techstock_new, investments
