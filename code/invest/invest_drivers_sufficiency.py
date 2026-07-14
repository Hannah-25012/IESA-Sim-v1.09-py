# File to identify the required driver stocks to satisfy economic drivers.

def invest_drivers_sufficiency(dimensions, activities, technologies, tech_stock_exist, report_gap, ip):

    # Extract parameters
    nA = dimensions['nA']
    driver_activities = [a for a in activities.entities if a.is_driver]

    # Identify sufficiency of driver stocks
    activity_gap = [0.0] * nA  # Preallocate

    if report_gap:
        print("---The remaining activity gaps are shown below:")
        print(f"{'Activity':>60},{'Gap':>6}")

    for activity in driver_activities:
        techs = activity.technologies

        activity_gap[activity.idx] = activity.volumes[ip] - sum(
            tech_stock_exist[t.idx] * t.cap2act for t in techs
        )

        if report_gap:
            print(f"{activity.name:>60},{activity_gap[activity.idx]:>6.2f}")

    return activity_gap
