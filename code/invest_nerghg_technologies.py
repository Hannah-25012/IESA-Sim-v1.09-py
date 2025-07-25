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
                    
                    if remaining_gap > 0:
                        tech_room = techstock_max[tech_coord].flatten() - techstock_exist[tech_coord].flatten() - valid_investments.flatten()
                        cand_availability = tech_room > 0
                        order_search = tech_choices_lcop_order[tech_coord]

                        order_idx = np.argsort(order_search)
                        fills = np.zeros(nT)
                        other_investments = np.zeros(nT)
                        rem = remaining_gap

                        for rank in order_idx:
                            if rem <= 0:
                                break
                            if cand_availability[rank]:
                                # invest as much as possible
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

    # To debug: Print the last 30 rows of the investments array
    # print(investments[-50:])

    return techstock_new, investments
