# Function to identify the required driver stocks to satisfy economic drivers.
# Attention: Code runs fine. Still need to check the for-loop - what the calculations do and whether they are correct. report_gap is defined as "false" in mod2_invest, not sure why?

def invest_drivers_sufficiency(dimensions, activities, technologies, tech_stock_exist, report_gap, ip):
   
    # Extract parameters
    nA = dimensions['nA']
    nAd = dimensions['nAd']
    activity_names = activities['names']
    activity_drivers = activities['drivers']['names']
    activity_net_volumes = activities['volumes'][:, ip]
    activity_per_tech = technologies['balancers']['activities']
    cap2act = technologies['balancers']['cap2acts']

    # Identify sufficiency of driver stocks
    activity_gap = [0.0] * nA  # Preallocate
    
    if report_gap:
        print("---The remaining activity gaps are shown below:")
        print(f"{'Activity':>60},{'Gap':>6}")

    for iAd in range(nAd):
        coord_act = [i for i, activity in enumerate(activity_names) if activity == activity_drivers[iAd]]
        coord_tech = [i for i, activity in enumerate(activity_per_tech) if activity == activity_names[coord_act[0]]]

        activity_gap[coord_act[0]] = activity_net_volumes[coord_act[0]] - sum(
            tech_stock_exist[i] * cap2act[i] for i in coord_tech
        )

        if report_gap:
            print(f"{activity_names[coord_act[0]]:>60},{activity_gap[coord_act[0]]:>6.2f}")

    return activity_gap
