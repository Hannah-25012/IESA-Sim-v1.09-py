# File to determine investments in non-energy related GHG saving technologies
import numpy as np
from invest_nerghg_sufficiency import invest_nerghg_sufficiency

def invest_nerghg_technologies(dimensions, activities, technologies,
                                techstock_exist, investment_potential, tech_choices, iP):

    # Extract Parameters
    nTb = dimensions['nTb']
    emission_activities = [a for a in activities.entities if a.is_emission]
    cap2act = technologies.balancers.cap2acts
    techstock_max = technologies.balancers.stocks.max[:, iP]
    tech_choices_lcop_order = technologies.balancers.choices_lcop_order[:, iP]

    # Initialize investments
    investments = np.zeros((nTb,1))

    # Determine the energy gap for current stock
    emission_gap = invest_nerghg_sufficiency(
        dimensions, activities, technologies, techstock_exist, True, iP
    )

    # Determine required investments
    for activity in emission_activities:

        # Check if the activity has an emission gap to fill
        if emission_gap[activity.idx] > 0:
            tech_coord = np.array([t.idx for t in activity.technologies if t.sector == 'nER GHG'], dtype=int)
            nT = len(tech_coord)

            # Check if there are nER-GHG technologies to invest in
            if nT > 0:
                cap2act_temp = cap2act[tech_coord]

                # Check max stock constraint
                choices_investments = emission_gap[activity.idx] * tech_choices[tech_coord] / cap2act_temp
                valid_investments = np.maximum(
                    np.minimum(investment_potential[tech_coord], choices_investments), 0
                )

                # Check sufficiency
                remaining_gap = emission_gap[activity.idx] - np.sum(valid_investments * cap2act_temp)

                # If gap remains, try to fill it
                if remaining_gap > 0:
                    tech_room = techstock_max[tech_coord].flatten() - techstock_exist[tech_coord].flatten() - valid_investments.flatten()
                    cand_availability = tech_room > 0
                    order_search = tech_choices_lcop_order[tech_coord]
                    order_idx = np.argsort(order_search)
                    fills = np.zeros(nT)
                    other_investments = np.zeros(nT)
                    rem = remaining_gap

                    # Fill the gap based on LCOP ranking
                    for rank in order_idx:
                        if rem <= 0:
                            break
                        if cand_availability[rank]:
                            possible = rem / cap2act_temp[rank]
                            amt = min(possible, tech_room[rank])
                            fills[rank] = amt
                            rem -= amt * cap2act_temp[rank]

                    # if still gap remains, assign to first technology
                    if rem > 0:
                        other_investments[0] = rem

                    # Sum all investments
                    investments[tech_coord, 0] = (
                        valid_investments.flatten() + fills + other_investments
                    )

    # Adjust investments and new stocks
    techstock_new = techstock_exist + investments

    return techstock_new, investments
