# File to determine investments in CCUS technologies to fill emission gaps
import numpy as np
from invest_ccus_sufficiency import invest_ccus_sufficiency

def invest_ccus_technologies(dimensions, activities, technologies,
                                techstock_exist, investment_potential, tech_choices, iP):

    # Extract Parameters
    nTb = dimensions['nTb']
    emission_activities = [a for a in activities.entities if a.is_emission]
    cap2act = technologies.balancers.cap2acts
    techstock_max = technologies.balancers.stocks.max[:, iP]

    # Initialize investments
    investments = np.zeros((nTb,1))

    # Determine the energy gap for current stock
    ccus_gap = invest_ccus_sufficiency(
        dimensions, activities, technologies, techstock_exist, False, iP
    )

    # Determine required investments
    for activity in emission_activities:

        # Check if the activity has an emission gap to fill
        if ccus_gap[activity.idx] > 0:
            tech_coord = np.array([t.idx for t in activity.technologies if t.subsector == 'CCUS'], dtype=int)
            nT = len(tech_coord)

            # Check if there are CCUS technologies to invest in
            if nT > 0:
                cap2act_temp = cap2act[tech_coord]

                # Check max stock constraint
                choices_investments = ccus_gap[activity.idx] * tech_choices[tech_coord] / cap2act_temp
                valid_investments = np.maximum(
                    np.minimum(investment_potential[tech_coord], choices_investments), 0
                )

                # Check sufficiency
                remaining_gap = ccus_gap[activity.idx] - np.sum(valid_investments * cap2act_temp)
                fill_investments = np.zeros((nT,1))  # Preallocate
                other_investments = np.zeros((nT,1))  # Preallocate

                # Fill the remaining gap if needed
                # NOTE: this branch is unreachable with real data (never observed
                # to trigger across this session's runs) and, as originally
                # written, would raise TypeError immediately if it ever did
                # (`iT = 1` is a plain int, then `len(iT)` below). Preserved
                # verbatim rather than silently "fixed", same as other
                # confirmed-dead branches found this session.
                if remaining_gap > 0:
                    tech_room = techstock_max[tech_coord].flatten() - techstock_exist[tech_coord].flatten() - valid_investments.flatten()
                    cand_availability = (tech_choices[tech_coord] > 0) * (tech_room > 0)
                    iT = 1

                    # Loop until the gap is filled or no more technologies are available
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

                        # Check if we have exceeded the number of technologies and the gap is not filled
                        if iT > nT:
                            print(f"!!!!Warning: There is not enough available max capacity to satisfy the demand for {activity.name}, the remaining gap is {-remaining_gap:6.2f} UoA") # Display a warning message
                            remaining_gap = 0

                    # Declare investments
                    valid_investments = valid_investments.reshape(-1, 1)
                    fill_investments = fill_investments.reshape(-1, 1)
                    other_investments = other_investments.reshape(-1, 1)
                    investments[tech_coord] = valid_investments + fill_investments + other_investments

    # Adjust investments and new stocks
    techstock_new = techstock_exist + investments
    return techstock_new, investments
