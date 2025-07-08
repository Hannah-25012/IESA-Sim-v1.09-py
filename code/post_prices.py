import numpy as np

def post_prices(dimensions, parameters, activities, technologies, policies, iP):
    # Extract Parameters
    nI = 5
    nH = dimensions['nH']
    nA = dimensions['nA']
    nAe = dimensions['nAe']
    nAc = dimensions['nAc']
    nTb = dimensions['nTb']
    nRp = dimensions['nRp']
    scarcity_penalization = parameters['scarcity']['penalization']
    activities_names = activities['names']
    activities_energy = activities['energies']['names']
    activities_emission = activities['emissions']['names']
    activity_label = activities['labels']
    activityType_act = activities['types']
    activity_resolution = activities['resolution']
    energy_scarcity = activities['energies']['scarcity'][:, iP]
    prices_hourly = activities['prices']['hourly'][:, :, iP]
    # reshaping to make sure the variable goed from shape (21,64,7) to (21,) as in Matlab
    energy_prices_ranges_full = activities['energies']['prices']['ranges']
    reshaped = energy_prices_ranges_full.reshape(dimensions['nRp'], -1)  
    energy_prices_ranges = reshaped[:, iP]

    price_ranges_hours = activities['energies']['prices']['price_ranges_hours']
    activityPer_tech = technologies['balancers']['activities']
    activity_balances = technologies['balancers']['activity_balances']
    tech_categories = technologies['balancers']['categories']
    vom_cost = technologies['balancers']['costs']['voms'][:, iP]
    cap2act = technologies['balancers']['cap2acts']
    tech_use = technologies['balancers']['use']['yearly'][:, iP]
    techUse_hourly = technologies['balancers']['use']['hourly'][:, :, iP]
    techStock = technologies['balancers']['stocks']['evolution'][:, iP]
    feedin_subject = technologies['balancers']['policies']['feedin_subject']
    feedin_activities = policies['feedins']['activities']
    feedin_values = policies['feedins']['values']
    taxes_activities = policies['taxes']['activities']
    taxes_values = policies['taxes']['values']
    
    # Energy and emissions coords (boolean arrays)
    coord_emission = np.array([atype == 'Emission' for atype in activityType_act])
    coord_energy = np.array([atype in ['Energy', 'Fix Energy'] for atype in activityType_act])
    
    # Preallocate cost arrays
    energy_prices_hourly_out = np.zeros((nH, nAe))
    emission_prices_hourly = np.zeros((nH, nAc))
    
    # Loop over iterations and activities
    for iI in range(nI):
        for iA in range(nA):
            # Identify the activity:
            # Create boolean arrays: True where the activity name matches in energies/emissions lists.
            iAe = np.array([act == activities_names[iA] for act in activities_energy])
            iAc = np.array([act == activities_names[iA] for act in activities_emission])
            
            if iAe.sum() + iAc.sum() > 0:
                # Check the temporal resolution of the dispatch.
                # In Matlab, check_not_yearly is the sum of comparisons; if zero then it is not yearly.
                check_not_yearly = activity_resolution[iA] in ['daily', 'hourly', 'hourly-interconnected']
                if not check_not_yearly:
                    # Select technologies that will be evaluated:
                    coord_tech = np.array([ap == activities_names[iA] for ap in activityPer_tech])
                    if coord_tech.sum() == 0:
                        # If no main technologies, use those with positive activity balance
                        coord_tech = activity_balances[:, iA] > 0
                    
                    nT = int(coord_tech.sum())
                    
                    # Calculate total production for the activity by these technologies.
                    # activity_balances[coord_tech, iA] becomes a 1D array (length nT)
                    # np.ones((nH,1)) @ ... creates an (nH x nT) matrix for multiplication.
                    total_prod = np.sum((techUse_hourly[:, coord_tech] + np.finfo(float).eps) *
                                          (np.ones((nH, 1)) @ activity_balances[coord_tech, iA].reshape(1, -1)),
                                          axis=1)  # shape: (nH,)
                    
                    # Determine production shares. If total production is zero overall, assume even use.
                    if total_prod.sum() > 0:
                        share_prod = (techUse_hourly[:, coord_tech] + np.finfo(float).eps) / (
                            total_prod.reshape(-1, 1) + np.finfo(float).eps)
                    else:
                        share_prod = np.ones((nH, nT)) / nT
                    
                    # Identify if there is a tax for this activity.
                    coord_taxes_act = np.array([act == activities_names[iA] for act in taxes_activities])
                    taxes_effect = 0
                    if coord_taxes_act.sum() > 0:
                        # If multiple, we use the first matching element.
                        idx_tax = np.where(coord_taxes_act)[0][0]
                        taxes_effect = taxes_values[idx_tax, iP]
                    
                    # Identify if there is a feed-in for this activity.
                    coord_feedin_act = np.array([act == activities_names[iA] for act in feedin_activities])
                    feedin_effect_tech = np.zeros(nT)
                    if coord_feedin_act.sum() > 0:
                        idx_feedin = np.where(coord_feedin_act)[0][0]
                        # Multiply the feed-in value by the element-wise product of activity balances and feedin_subject
                        feedin_effect_tech = feedin_values[idx_feedin, iP] * (
                            activity_balances[coord_tech, iA] * feedin_subject[coord_tech])
                    # Ensure nonnegative feedin effect
                    feedin_effect_tech = np.maximum(feedin_effect_tech, 0)
                    
                    # Obtain the production costs per technology (nH x nT)
                    cost_prod = np.zeros((nH, nT))
                    vom_cost_temp = vom_cost[coord_tech]  # length nT
                    # Use boolean indexing to select rows for energy and emission balances.
                    energy_balances_temp = activity_balances[coord_tech][:, coord_energy]  # size: nT x (num energy coords)
                    emission_balances_temp = activity_balances[coord_tech][:, coord_emission]  # size: nT x (num emission coords)
                    
                    # Loop over each technology index for this activity
                    for iT in range(nT):
                        # Identify activities with positive energy balances for this technology.
                        energy_pos = energy_balances_temp[iT, :] > 0  # boolean mask
                        
                        # Identify cogeneration share (computed but not used later)
                        if iAe.sum() > 0:
                            # Here Matlab uses energy_balances_temp(iT, iAe) which in Matlab returns all matching columns.
                            # For translation, we take the first True index.
                            idx_energy = np.where(iAe)[0][0]
                            denom = np.sum(energy_balances_temp[iT, energy_pos])
                            co_share = (energy_balances_temp[iT, idx_energy] / denom) if denom != 0 else 0
                        else:
                            co_share = 1
                            # Set the emission balance corresponding to the first True in iAc to zero.
                            idx_emiss = np.where(iAc)[0][0]
                            emission_balances_temp[iT, idx_emiss] = 0
                        
                        # Modify the balances to include only consumed energy:
                        energy_balances_temp[iT, energy_pos] = 0
                        
                        # Calculate production costs:
                        # Subtract variable costs, energy costs, emission costs, feedin subsidy and add taxes.
                        cost_prod[:, iT] = (
                            vom_cost_temp[iT]
                            - np.sum(energy_prices_hourly_out * (np.ones((nH, 1)) @ energy_balances_temp[iT, :].reshape(1, -1)), axis=1)
                            - np.sum(emission_prices_hourly * (np.ones((nH, 1)) @ emission_balances_temp[iT, :].reshape(1, -1)), axis=1)
                            - feedin_effect_tech[iT]
                            + taxes_effect
                        )
                    
                    # Split the production costs per tech use share.
                    if iAe.sum() > 0:
                        # In Matlab, energy_prices_hourly(:,iAe) updates all columns corresponding to True.
                        # In Python, use boolean indexing.
                        energy_prices_hourly_out[:, iAe] = np.sum(cost_prod * share_prod, axis=1, keepdims=True)
                    else:
                        emission_prices_hourly[:, iAc] = np.sum(cost_prod * share_prod, axis=1, keepdims=True)
                
                else:
                    # For the "yearly" resolution: if first iteration (iI==0 in Python equals iI==1 in Matlab) and energy
                    if iI == 0 and iAe.sum() > 0:
                        energy_prices_hourly_out[:, iAe] = prices_hourly[:, iA, None]

    # Obtain the yearly average prices
    energy_prices = np.zeros(nAe)
    for iAe in range(nAe):
        energy_prices[iAe] = np.sum(energy_prices_hourly_out[:, iAe]) / nH

    emission_prices = np.zeros(nAc)
    for iAc in range(nAc):
        emission_prices[iAc] = np.sum(emission_prices_hourly[:, iAc]) / nH

    # Adjust prices accordingly with the scarcity price
    # Obtain the price curve for all primary energy technologies 
    # that are not Electricity
    primary_available = np.zeros(nTb)
    primary_price = np.zeros(nTb)
    for iTb in range(nTb):
        # Find the activity index matching the current technology's activity
        if activityPer_tech[iTb] in activities_names:
            act_coord = activities_names.index(activityPer_tech[iTb])
        else:
            act_coord = None

        # Check if the technology is 'Primary' and its associated activity is not Electricity
        if (tech_categories[iTb] == 'Primary') and (act_coord is not None) and (activity_label[act_coord] != 'Electricity'):
            primary_available[iTb] = techStock[iTb] * cap2act[iTb] - tech_use[iTb]
            primary_price[iTb] = vom_cost[iTb]

    # Sort primary prices in ascending order and compute cumulative volume
    order = np.argsort(primary_price)
    primary_curve_price = primary_price[order]
    primary_curve_volume = np.cumsum(primary_available[order])

    # Obtain the scarcity price based on total scarcity
    total_scarcity = np.sum(energy_scarcity)
    # In MATLAB: scarcity_coord = min(sum((primary_curve_volume < total_scarcity) + 1), length(primary_curve_price));
    scarcity_coord = min(int(np.sum(primary_curve_volume < total_scarcity)) + 1, len(primary_curve_price))
    scarcity_price = primary_curve_price[scarcity_coord - 1]

    # Check all energy activities for scarcity adjustment
    for iAe in range(nAe):
        # Determine the corresponding activity index for this energy activity
        if activities_energy[iAe] in activities_names:
            act_index = activities_names.index(activities_energy[iAe])
        else:
            act_index = None
        # Check the temporal resolution of the dispatch:
        # (In MATLAB, check_not_yearly equals the sum of strcmp results and is zero when resolution is yearly)
        check_not_yearly = False
        if act_index is not None:
            if activity_resolution[act_index] in ['daily', 'hourly', 'hourly-interconnected']:
                check_not_yearly = True
        # If the dispatch resolution is yearly (i.e. check_not_yearly is False) and scarcity is present...
        if not check_not_yearly:
            if energy_scarcity[iAe] > 0:
                old_price = energy_prices[iAe]
                energy_prices[iAe] = old_price + abs(old_price - scarcity_price) * scarcity_penalization
                print(f'.... Energy price of {activities_energy[iAe]} was increased from {old_price:.2f} EUR/GJ to {energy_prices[iAe]:.2f} EUR/GJ due to scarcity.')

    # Obtain price ranges distribution for the hourly prices
    energy_prices_ranges = np.empty((nRp, nAe))

    for iAe in range(nAe):
        sorted_prices = np.sort(energy_prices_hourly_out[:, iAe])
        for iRp in range(nRp):
            nHr = price_ranges_hours[iRp]
            energy_prices_ranges[iRp, iAe] = np.mean(sorted_prices[:int(nHr)])

    # Save Variables
    activities['energies']['prices']['yearly'][:, iP] = energy_prices
    # Note: seems correct, still have to check the sum
    activities['energies']['prices']['hourly'][:, :, iP] = energy_prices_hourly_out
    activities['energies']['prices']['ranges'][:, :, iP] = energy_prices_ranges
    activities['emissions']['prices']['yearly'][:, iP] = emission_prices

    return activities

