import numpy as np

def initialize_prices(activities, tech_categories, activity_label, activity_per_tech,
                      activities_energy, activities_emission, taxes_activities,
                      taxes_values, vom_cost):


    activity_names = activities['names']
    nA = len(activity_names)

    # Preallocate initialized prices array
    initialized_prices = np.zeros((nA, vom_cost.shape[1]))

    for iA in range(nA):
        # Handle primary energy sources
        if activity_names[iA] in activities_energy:
            coord_act = [label == activity_label[iA] for label in activity_label]
            nAs = sum(coord_act)
            activities_similar = [activity_names[i] for i, val in enumerate(coord_act) if val]

            primary_cost = np.zeros((10, vom_cost.shape[1]))  # Preallocate
            iT = 0

            for iAs in range(nAs):
                coord_tech = [(activity_per_tech[j] == activities_similar[iAs] and tech_categories[j] == 'Primary')
                              for j in range(len(activity_per_tech))]
                nT = iT + sum(coord_tech)
                primary_cost[iT:nT, :] = vom_cost[coord_tech, :]
                iT = nT

            if iT > 0:
                initialized_prices[iA, :] = np.sum(primary_cost, axis=0) / iT

        # Handle emission-related activities
        if activity_names[iA] in activities_emission:
            coord_tech = [activity_per_tech[j] == activity_names[iA] for j in range(len(activity_per_tech))]
            initialized_prices[iA, :] = np.max(vom_cost[coord_tech, :], axis=0)

        # Adjust for taxes
        coord_taxes_act = [activity_names[iA] == tax_act for tax_act in taxes_activities]
        coord_taxes_act = np.array(coord_taxes_act, dtype=bool)
        if sum(coord_taxes_act) > 0:
            taxes_effect = taxes_values[coord_taxes_act, :]
            #to squeeze the variable taxes_effect into a list
            taxes_effect_list = taxes_effect.squeeze().tolist()
            initialized_prices[iA, :] -= taxes_effect_list

    return initialized_prices