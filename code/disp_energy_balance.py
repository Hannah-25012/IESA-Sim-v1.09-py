import numpy as np
# function to match matlab rounding behavior - though this is not up to date according to ChatGPT, modern matlab uses round‐half‐to‐even (banker’s rounding)
# from disp_energy_balance_round import round_half_away_from_zero

def disp_energy_balance(dimensions, activities, technologies, profiles, tech_use_hourly, report_gap, iP):
    # Define printing precision to compare with Matlab
    np.set_printoptions(precision=15)
    
    # Export Parameters
    nIb = dimensions['nIb'] # correct
    nA = dimensions['nA'] # correct
    nTb = dimensions['nTb'] # correct
    activities_names = activities['names']
    activity_resolution = activities['resolution']
    activities_net_volumes = activities['volumes'][:, iP] # correct
    activity_per_tech = technologies['balancers']['activities']
    activity_balances = technologies['balancers']['activity_balances'] # correct
    tech_stock = technologies['balancers']['stocks']['evolution'][:, iP] # correct
    tech_categories = technologies['balancers']['categories']
    cap2act = technologies['balancers']['cap2acts'] # almost correct: Python: 6925.3842, matlab: 6.925384200000006e+03
    hourly_profile_tech = technologies['balancers']['profiles']
    shedding_capacity = technologies['balancers']['shedding']['capacity']
    profile_type = profiles['types']
    hourly_profiles = profiles['shapes'] # almost correct: Python: 51.00000004000006, matlab: 5.100000004000005e+01

    # Iterate in a loop
    tech_use = np.sum(tech_use_hourly, axis=0)
    tech_use_delta = np.zeros((nTb, nIb))  # Preallocate (in units of activity)

    for iI in range(nIb):
        for iA in range(nA):
            # Do the loop for all activities except hourly and daily
            if all(activity_resolution[iA] != resolution for resolution in ['daily', 'hourly']):

                # Identify the remaining gap
                # Note: act_balance is the variable for which the discrepancy starts. I don't know why. Small difference in the last few decimals.
                # act_balance = round_half_away_from_zero((tech_use[:, np.newaxis] * np.ones((1, nA))) * activity_balances,14)  # ntb x na  # ntb x na
                # act_balance = (tech_use[:, np.newaxis] * np.ones((1, nA))) * activity_balances  # ntb x na  # ntb x na

                # act_balance = round_half_away_from_zero(np.outer(tech_use, np.ones(nA)) * activity_balances,14)  # ntb x na  # ntb x na
                # act_balance = np.round(np.outer(tech_use, np.ones(nA)) * activity_balances, 14) 

                act_balance = tech_use[:, None] * activity_balances
 

                # debugging
                print("iteration", iI, "activity", iA)
                print("total act_balance_sum:", act_balance.sum())

                activity_gap = activities_net_volumes[iA] - np.sum(act_balance[:, iA])
                # debugging
                print("activity_gap:", activity_gap)

                # Calculate the remaining use space of technologies
                tech_use_max = tech_stock * cap2act
                rem_use = np.maximum(tech_use_max - tech_use, 0)

                # Identify the involved technologies
                coord_tech_act = np.array(activity_per_tech) == activities_names[iA]
                coord_tech_prim = np.array(tech_categories) == 'Primary'
                coord_tech_shed = shedding_capacity > 0

                coord_i3 = coord_tech_act & ~coord_tech_prim & ~coord_tech_shed
                coord_i4 = coord_tech_act & ~coord_tech_shed
                
                if iI > 2:  
                    coord = coord_i4
                else:
                    coord = coord_i3
                

                # Calculate the use factor of each technology in the selection
                if activity_gap > 0:
                    use_factor = rem_use[coord] / np.sum(rem_use[coord] + 1e-9)


                else:
                    use_factor = tech_use[coord] / np.sum(tech_use[coord] + 1e-9)


                # Adjust tech_use of the iteration
                tech_use_delta[coord, iI] = activity_gap * use_factor
                # debugging
                # print("sum of tech_use_delta:", np.sum(tech_use_delta, axis=0))

                tech_use = tech_use.copy()
                tech_use_delta = tech_use_delta.copy()
                tech_use_max = tech_use_max.copy()

                tech_use[coord] = np.maximum(np.minimum(tech_use_max[coord], tech_use[coord] + tech_use_delta[coord, iI]), 0)
                # debugging
                # print("tech_use[coord]", tech_use[coord])

                tech_use = tech_use.copy()

                if iI == nIb - 1:
                    tech_sel = np.where(coord_tech_act)[0]
                    nT = np.sum(coord_tech_act)
                    for iT in range(nT):
                        coord_profile = np.array(profile_type) == hourly_profile_tech[tech_sel[iT]]
                        tech_use_hourly[:, tech_sel[iT]] = tech_use[tech_sel[iT]] * hourly_profiles[:, coord_profile].flatten()
                        
        # debugging
        print("iteration:", iI, "sum of tech_use:", np.sum(tech_use))
        
    tech_use = (np.sum(tech_use_hourly, axis=0))
    # debugging
    print("sum of tech_use:", np.sum(tech_use))

    # Report the remaining energy gaps
    if report_gap:
        # Calculate the gaps
        act_balance = (tech_use[:, np.newaxis] * np.ones((1, nA))) * activity_balances  # ntb x na
        activity_gap = activities_net_volumes - np.sum(act_balance, axis=0)

        # Print the activity gaps
        print(f"{'Activity':>60},{'Gap':>6}")
        for iA in range(nA):
            print(f"{activities_names[iA]:>60},{activity_gap[iA]:6.2f}")
            if activity_resolution[iA] in ('daily', 'hourly'):
                pos_balance = act_balance[:, iA] > 0
                print(f"*Total co-generated in processes: {np.sum(act_balance[pos_balance, iA]):6.2f}")

    return tech_use, tech_use_hourly