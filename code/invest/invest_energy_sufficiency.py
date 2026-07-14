# File to determine the required installed capacities for energy
import numpy as np

def invest_energy_sufficiency(dimensions, activities, technologies, techstock_exist, report_gap, iP):

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

    # debugging: ensure each outcome is rounded to exactly the same number of decimals (same is implemented in matlab)
    precision = 14 # define rounding precision
    intermediate_result = techstock_test * cap2act
    intermediate_result = np.round(intermediate_result, decimals=precision)
    actbalance_max = intermediate_result @ np.ones((1, nA))
    actbalance_max = np.round(actbalance_max, decimals=precision)
    actbalance_max *= activity_balances
    actbalance_max = np.round(actbalance_max, decimals=precision)

    # Calculate the gaps
    activity_gap = activities_netvolumes - np.sum(actbalance_max, axis=0)
    activity_gap = np.round(activity_gap, decimals=precision)
    activity_gap[activity_gap < 0] = 0

    # Extract only the energy gaps
    energy_gap = np.zeros((nA, 1))

    if report_gap:
        print(f"{'Activity':>60s}, {'Gap':>6s}")

    for activity in activities.entities:
        if activity.is_energy:
            energy_gap[activity.idx] = activity_gap[activity.idx]

            if report_gap:
                print(f"{activity.name:>60s}, {energy_gap[activity.idx][0]:6.2f}")

    return energy_gap
