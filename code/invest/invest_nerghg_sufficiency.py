# File to determine the required installed capacities for nEr GHG emission saving technologies
import numpy as np

def invest_nerghg_sufficiency(dimensions, activities, technologies, techstock_exist, report_gap, iP):

    # Extract Parameters
    nA = dimensions['nA']
    activities_netvolumes = activities.volumes[:, iP]
    tech_entities = technologies.balancers.entities
    activity_balances = technologies.balancers.activity_balances

    # Calculate the activity balance of existing stock without electricity generation
    techstock_test = techstock_exist.copy()
    for tech in tech_entities:
        if tech.activity is not None and tech.activity.is_electricity:
            techstock_test[tech.idx] = 0

    cap2act = technologies.balancers.cap2acts.reshape(-1, 1)
    actbalance_max = (techstock_test * cap2act) @ np.ones((1, nA)) * activity_balances # ntb x na

    # Calculate the gaps
    activity_gap = activities_netvolumes - np.sum(actbalance_max, axis=0)
    activity_gap[activity_gap < 0] = 0

    # Extract only the emission gaps
    emission_gap = np.zeros((nA, 1))

    if report_gap:
        print(f"{'Activity':>60s}, {'Gap':>6s}")

    for activity in activities.entities:
        if activity.is_emission:
            emission_gap[activity.idx] = activity_gap[activity.idx]

            if report_gap:
                print(f"{activity.name:>60s}, {emission_gap[activity.idx][0]:6.2f}")

    return emission_gap
