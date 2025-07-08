import numpy as np

def invest_ccus_sufficiency(dimensions, activities, technologies, techstock_exist, report_gap, iP):
    
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

    # debugging: ensure each outcome is rounded to exactly the same number of decimals (same is implemented in matlab)
    # define rounding precision
    precision = 14
    intermediate_result = techstock_test * cap2act
    intermediate_result = np.round(intermediate_result, decimals=precision)
    # debugging: printing intermediate outcomes
    # print("Intermediate result (techstock_test * cap2act):", intermediate_result)

    actbalance_max = intermediate_result @ np.ones((1, nA))
    actbalance_max = np.round(actbalance_max, decimals=precision)

    actbalance_max *= activity_balances
    actbalance_max = np.round(actbalance_max, decimals=precision)
 
    # actbalance_max = (techstock_test * cap2act) @ np.ones((1, nA)) * activity_balances # ntb x na

    # Calculate the gaps
    # Attention: for activity no. 127 (python) or 128 (matlab), numbers don't match. Activity gap is incorrect, so ccus_gap is incorrect, too. I don't understand how this happens
    # calculation seems to be right, need to double-check.
    activity_gap = activities_netvolumes - np.sum(actbalance_max, axis=0)
    activity_gap = np.round(activity_gap, decimals=precision)
    # Manuel: In module invest_nergh_sufficiency, it's <0 --> is this correct?
    activity_gap[activity_gap > 0] = 0

    # Extract only the emission gaps
    ccus_gap = np.zeros((nA, 1))
    if report_gap:
        print(f"{'Activity':>60s},{'Gap':>6s}")

    for iAc in range(nAc):
        coord_act = np.array([act == activities_emission[iAc] for act in activities_names])
        ccus_gap[coord_act] = activity_gap[coord_act]
        if report_gap:
            indices = np.where(coord_act)[0]
        
            if indices.size != 1:
                raise ValueError(f"Unexpected number of matches ({indices.size}) for activities_emission[{iAc}].")
        
            # Use the scalar index to access data for printing
            index = indices[0]
            print(f"{activities_names[index]:>60s}, {ccus_gap[index][0]:6.2f}")
    
    # Attention: ccus_gap for activity number 127 incorrect, different to matlab no. 128! Rounding also doesn't fix the issue... Need to check what's up here.
    return ccus_gap