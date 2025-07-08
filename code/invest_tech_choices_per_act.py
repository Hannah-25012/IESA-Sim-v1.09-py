import numpy as np

def invest_tech_choices_per_act(dimensions, activities, technologies, agents, iP):
    # Extract parameters
    nA = dimensions['nA']
    nTb = dimensions['nTb']
    nAT = dimensions['nAT']
    activities_names = activities['names']
    activity_agent = activities['agents']
    activity_per_tech = technologies['balancers']['activities']
    dispatch_type_tech = technologies['balancers']['dispatch']
    tech_LCOPs = technologies['balancers']['lcops']['values'][:, iP]
    multi_criteria_performance_tech = technologies['balancers']['mca']['matrix'][:, :, iP]
    tech_choices_agent = technologies['balancers']['choices_agent'][:, :, iP]
    agent_profiles = agents['profiles']
    weights_multicriteria = agents['criteria']['weights']
    agents_populations = agents['populations']
    tech_choices_LCOP_order = technologies['balancers']['choices_lcop_order']

    # Add a large cost to buffer technologies
    coord_buffer = np.array([dt == 'Gas buffer' for dt in dispatch_type_tech])
    tech_LCOPs[coord_buffer] += 1e6

    # Initialize LCOP order choices
    # tech_choices_LCOP_order = np.zeros(nTb, dtype=int)

    for iA in range(nA):
        # Identify technologies that satisfy the activity
        coord_tech_act = np.array([activities_names[iA] == act for act in activity_per_tech])
        nTa = np.sum(coord_tech_act)

        # Order technologies by LCOP
        tech_LCOPs_options = tech_LCOPs[coord_tech_act]
        order_LCOP = np.argsort(tech_LCOPs_options)
        order_in_line = np.zeros(nTa, dtype=int)
        for iTa in range(nTa):
            order_in_line[order_LCOP[iTa]] = iTa + 1
        tech_choices_LCOP_order[np.where(coord_tech_act)[0], iP] = order_in_line

        # Identify non-buffer technologies
        coord_tech_opr = ~coord_buffer

        # Identify population vector based on agent profiles
        coord_agentprofile = np.array([activity_agent[iA] == ap for ap in agent_profiles])
        population_vector = agents_populations[coord_agentprofile, :].flatten()

        # Check for main technologies
        coord_tech = coord_tech_act & coord_tech_opr
        nT = np.sum(coord_tech)
        if nT == 0:
            print(f"--****There is no main technology for activity: {activities_names[iA]}")
        elif nT == 1:
            tech_choices_agent[coord_tech, :] = population_vector
        else:
            # Extract multicriteria of technologies
            multicriteria_matrix = multi_criteria_performance_tech[coord_tech, :]

            # Multi-criteria valuation per agent type
            multicriteria_valuation = np.zeros((nT, nAT))
            for iAT in range(nAT):
                multicriteria_valuation[:, iAT] = np.sum(
                    multicriteria_matrix * weights_multicriteria[:, iAT], axis=1
                )
                coord_max = multicriteria_valuation[:, iAT] == np.max(multicriteria_valuation[:, iAT])
                nT_max = np.sum(coord_max)
                tech_choices_inter = np.zeros(nT)
                if nT_max > 1:
                    multicriteria_CO2 = multicriteria_matrix[coord_max, 1]
                    multicriteria_LCOP = multicriteria_matrix[coord_max, 2]
                    # Better LCOP
                    coord_LCOP = multicriteria_LCOP == np.max(multicriteria_LCOP)
                    nT_LCOP = np.sum(coord_LCOP)
                    tech_choices_inter_inter = np.zeros(nT_max)
                    if nT_LCOP > 1:
                        multicriteria_CO2 = multicriteria_CO2[coord_LCOP]
                        # Better environmental footprint
                        coord_CO2 = multicriteria_CO2 == np.max(multicriteria_CO2)
                        nT_CO2 = np.sum(coord_CO2)
                        tech_choices_inter_inter_inter = np.zeros(nT_LCOP)
                        # Proportional
                        # Note: What does proportional mean here?
                        tech_choices_inter_inter_inter[coord_CO2] = (
                            population_vector[iAT] / nT_CO2
                        )
                        tech_choices_inter_inter[coord_LCOP] = tech_choices_inter_inter_inter
                    else:
                        tech_choices_inter_inter[coord_LCOP] = population_vector[iAT]

                    tech_choices_inter[coord_max] = tech_choices_inter_inter
                else:
                    tech_choices_inter[coord_max] = population_vector[iAT]

                tech_choices_agent[coord_tech, iAT] = tech_choices_inter

    tech_choices = np.sum(tech_choices_agent, axis=1)

    # Save variables
    technologies['balancers']['choices_agent'][:, :, iP] = tech_choices_agent
    technologies['balancers']['choices_lcop_order'][:, iP] = tech_choices_LCOP_order[:, iP]

    return technologies, tech_choices