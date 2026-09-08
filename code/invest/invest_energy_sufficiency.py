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

    # OPTION B: don't let the gap that drives force-investment (below, in
    # invest_energy_technologies.py) chase more NEW capacity than an
    # infra-linked activity's own delivery network could ever carry. Without
    # this, e.g. Hydrogen HD's exogenous demand target alone decides how
    # much electrolyzer capacity gets force-built, with no awareness that
    # the pipe delivering it (infrastructure_stocks.max) has its own,
    # separate ceiling - see disp_infrastructure.py, which only bounds what
    # gets REPORTED as built infra, not what invests upstream of it.
    #
    # infra_obj.cap2act converts GW of pipe -> PJ/DAY (matches
    # disp_infrastructure.py's own peak-daily-flow usage); activities_
    # netvolumes/activity_gap here are annual PJ, so *365 assumes the daily
    # peak could sustain all year - the most permissive (least risky)
    # annualisation of a daily figure, not an average-load assumption.
    for activity in activities.entities:
        if not activity.infrastructure:
            continue
        infra_obj = activity.infrastructure[0]
        if infra_obj.cap2act == 0:
            continue
        infra_capacity_annual = infra_obj.stock_max[iP] * infra_obj.cap2act * 365
        already_supplied = actbalance_max[:, activity.idx].sum()
        allowed_gap = max(0.0, infra_capacity_annual - already_supplied)
        if activity_gap[activity.idx] > allowed_gap:
            activity_gap[activity.idx] = allowed_gap

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
