# File to determine actors' technology choices
import numpy as np

def invest_tech_choices_per_act(dimensions, activities, technologies, agents, iP):

    # Extract parameters
    nTb = dimensions['nTb']
    nAT = dimensions['nAT']
    tech_entities = technologies.balancers.entities
    tech_LCOPs = technologies.balancers.lcops.values[:, iP]
    multi_criteria_performance_tech = technologies.balancers.mca.matrix[:, :, iP]
    tech_choices_agent = technologies.balancers.choices_agent[:, :, iP]
    agent_profiles = agents.profiles
    weights_multicriteria = agents.criteria.weights
    agents_populations = agents.populations

    # Agent-profile name -> row lookup, resolved once instead of re-scanning
    # agent_profiles for every activity.
    agent_idx_by_name = {name: i for i, name in enumerate(agent_profiles)}

    # Add a large cost to buffer technologies (to make them unattractive and thereby not chosen)
    coord_buffer = np.array([tech.is_buffer for tech in tech_entities])
    tech_LCOPs[coord_buffer] += 1e6

    # Identify choice per activity and agent type
    tech_choices_LCOP_order = np.zeros(nTb, dtype=int)
    for activity in activities.entities:

        # Identify technologies that satisfy the activity
        idxs = np.array([t.idx for t in activity.technologies], dtype=int)
        nTa = idxs.size

        # Order technologies by LCOP
        tech_LCOPs_options = tech_LCOPs[idxs]
        order_LCOP = np.argsort(tech_LCOPs_options, kind='mergesort')
        order_in_line = np.zeros(nTa, dtype=int)
        for iTa in range(nTa):
            order_in_line[order_LCOP[iTa]] = iTa + 1

        tech_choices_LCOP_order[idxs] = order_in_line

        # Identify non-buffer technologies for this activity
        competitors = [t for t in activity.technologies if not t.is_buffer]
        coord_tech = np.array([t.idx for t in competitors], dtype=int)
        nT = len(competitors)

        # Identify population vector of agent types based on the activity's agent profile
        agent_idx = agent_idx_by_name.get(activity.agent_profile_name)
        population_vector = agents_populations[agent_idx, :]

        if nT == 0:
            print(f"--****There is no main technology for activity: {activity.name}")

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

                # If there are multiple best technologies, check for better LCOP or better environmental footprint
                # THINK: This prioritises costs first and then CO2. Social and complexity not considered
                if nT_max > 1:
                    multicriteria_CO2 = multicriteria_matrix[coord_max, 1]
                    multicriteria_LCOP = multicriteria_matrix[coord_max, 2]

                    # Choose better LCOP
                    coord_LCOP = multicriteria_LCOP == np.max(multicriteria_LCOP)
                    nT_LCOP = np.sum(coord_LCOP)
                    tech_choices_inter_inter = np.zeros(nT_max)

                    # If more than one with better LCOP, choose better environmental footprint
                    if nT_LCOP > 1:
                        multicriteria_CO2 = multicriteria_CO2[coord_LCOP]
                        coord_CO2 = multicriteria_CO2 == np.max(multicriteria_CO2)
                        nT_CO2 = np.sum(coord_CO2)
                        tech_choices_inter_inter_inter = np.zeros(nT_LCOP)
                        # Proportional
                        # UNDERSTAND: What does proportional mean here?
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
    technologies.balancers.choices_agent[:, :, iP] = tech_choices_agent
    technologies.balancers.choices_lcop_order[:, iP] = tech_choices_LCOP_order

    return technologies, tech_choices
