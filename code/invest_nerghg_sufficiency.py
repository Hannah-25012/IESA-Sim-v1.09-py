# File to determine the required installed capacities for energy

import numpy as np

def invest_nerghg_sufficiency(dimensions, activities, technologies, techstock_exist, report_gap, iP):
    
    # Extract Parameters
    nA = dimensions['nA']
    nAc = dimensions['nAc']
    nTb = dimensions['nTb']
    activities_names = activities['names']
    activity_label = activities['labels']
    activities_emission = activities['emissions']['names']
    activities_netvolumes = activities['volumes'][:, iP]
    activity_per_tech = technologies['balancers']['activities']
    cap2act = technologies['balancers']['cap2acts']
    activity_balances = technologies['balancers']['activity_balances']

    # Calculate the activity balance of existing stock without electricity generation
    techstock_test = techstock_exist.copy()
    for iTb in range(nTb):
        coord_act = np.array([act == activity_per_tech[iTb] for act in activities_names])

        activity_label = np.array(activity_label)

        matched_labels = activity_label[coord_act]
        if matched_labels.size > 0 and matched_labels[0] == 'Electricity':
            techstock_test[iTb] = 0

    
    cap2act = cap2act.reshape(-1, 1)
 
    actbalance_max = (techstock_test * cap2act) @ np.ones((1, nA)) * activity_balances # ntb x na


    # Calculate the gaps
    activity_gap = activities_netvolumes - np.sum(actbalance_max, axis=0)
    activity_gap[activity_gap < 0] = 0

    # Extract only the emission gaps
    emission_gap = np.zeros((nA, 1))
    if report_gap:
        print(f"{'Activity':>60s}, {'Gap':>6s}")

    for iAc in range(nAc):
        coord_act = np.array([act == activities_emission[iAc] for act in activities_names])
        emission_gap[coord_act] = activity_gap[coord_act]
        if report_gap:
            indices = np.where(coord_act)[0]
        
            if indices.size != 1:
                raise ValueError(f"Unexpected number of matches ({indices.size}) for activities_emission[{iAc}].")
        
            # Use the scalar index to access data for printing
            index = indices[0]
            print(f"{activities_names[index]:>60s}, {emission_gap[index][0]:6.2f}")

    return emission_gap
