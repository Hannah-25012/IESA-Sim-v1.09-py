# Fiel to postprocess the energy prices in the solution
import numpy as np

def post_prices(dimensions, parameters, activities, technologies, policies, iP):

    # Extract Parameters
    nI = 5
    nH = dimensions['nH']
    nAe = dimensions['nAe']
    nAc = dimensions['nAc']
    nRp = dimensions['nRp']
    nTb = dimensions['nTb']
    scarcity_penalization = parameters.scarcity.penalization
    activity_entities = activities.entities
    energy_activities = [a for a in activity_entities if a.is_energy]
    tech_entities = technologies.balancers.entities
    energy_scarcity = activities.energies.scarcity[:, iP]
    prices_hourly = activities.prices.hourly[:, :, iP]
    energy_prices_ranges_full = activities.energies.prices.ranges
    reshaped = energy_prices_ranges_full.reshape(dimensions['nRp'], -1) # reshaping to make sure the variable go from shape (21,64,7) to (21,) as in Matlab
    energy_prices_ranges = reshaped[:, iP]
    price_ranges_hours = activities.energies.prices.price_ranges_hours
    # Emission-type activities are now stored positive-for-emitters (IESA-Opt
    # convention). This function prices emission activities too (not just
    # energy ones), including cases where the activity being priced IS itself
    # emission-type (e.g. 'nER-GHG CO2') - every raw use of activity_balances
    # below, not just the emission_balances_temp slice, needs the old
    # (negative-for-emitters) convention. Restore it once, up front, on this
    # local copy rather than patching each call site individually.
    activity_balances = technologies.balancers.activity_balances.copy()
    _coord_emission_act = np.array([a.is_emission for a in activities.entities])
    activity_balances[:, _coord_emission_act] = -activity_balances[:, _coord_emission_act]
    vom_cost = technologies.balancers.costs.voms[:, iP]
    cap2act = technologies.balancers.cap2acts
    tech_use = technologies.balancers.use.yearly[:, iP]
    techUse_hourly = technologies.balancers.use.hourly[:, :, iP]
    techStock = technologies.balancers.stocks.evolution[:, iP]
    feedin_subject = technologies.balancers.policies.feedin_subject
    feedin_activities = policies.feedins.activities
    feedin_values = policies.feedins.values
    taxes_activities = policies.taxes.activities
    taxes_values = policies.taxes.values

    # Activity-name -> period-column lookups, resolved once instead of
    # re-scanning the policy activity lists for every activity (matching the
    # pattern established in invest_tech_LCOPs.py).
    taxes_idx_by_name = {name: i for i, name in enumerate(taxes_activities)}
    feedin_idx_by_name = {name: i for i, name in enumerate(feedin_activities)}

    # Energy and emissions coords (boolean masks over ALL activities)
    coord_emission = np.array([a.is_emission for a in activity_entities])
    coord_energy = np.array([a.is_energy for a in activity_entities])

    # Solve iterative loop to quantify production costs
    energy_prices_hourly_out = np.zeros((nH, nAe)) # Preallocate
    emission_prices_hourly = np.zeros((nH, nAc))
    for iI in range(nI):
        for activity in activity_entities:

            # If the activity is energy or emission-related
            if activity.energy_idx is not None or activity.emission_idx is not None:

                # Check the temporal resolution of the dispatch.
                check_not_yearly = activity.resolution in ['daily', 'hourly', 'hourly-interconnected'] # In Matlab, check_not_yearly is the sum of comparisons; if zero then it is not yearly.
                if not check_not_yearly:

                    # Select technologies that will be evaluated:
                    coord_tech = np.zeros(nTb, dtype=bool)
                    if activity.technologies:
                        coord_tech[[t.idx for t in activity.technologies]] = True

                    # If no main technologies, use those with positive activity balance
                    if coord_tech.sum() == 0:
                        coord_tech = activity_balances[:, activity.idx] > 0

                    # Identify total production of the activity by those technologies and their shares
                    nT = int(coord_tech.sum())
                    total_prod = np.sum((techUse_hourly[:, coord_tech] + np.finfo(float).eps) * # activity_balances[coord_tech, iA] becomes a 1D array (length nT)
                                          (np.ones((nH, 1)) @ activity_balances[coord_tech, activity.idx].reshape(1, -1)), # np.ones((nH,1)) @ ... creates an (nH x nT) matrix for multiplication.
                                          axis=1)  # shape: (nH,)

                    # Identify if any technology was in use. Otherwise assume an even use of technology
                    if total_prod.sum() > 0:
                        share_prod = (techUse_hourly[:, coord_tech] + np.finfo(float).eps) / (
                            total_prod.reshape(-1, 1) + np.finfo(float).eps)
                    else:
                        share_prod = np.ones((nH, nT)) / nT

                    # Identify if there is a tax for this activity.
                    taxes_effect = 0
                    if activity.name in taxes_idx_by_name:
                        idx_tax = taxes_idx_by_name[activity.name]
                        taxes_effect = taxes_values[idx_tax, iP]

                    # Identify if there is a feed-in for this activity.
                    feedin_effect_tech = np.zeros(nT)
                    if activity.name in feedin_idx_by_name:
                        idx_feedin = feedin_idx_by_name[activity.name]
                        feedin_effect_tech = feedin_values[idx_feedin, iP] * (
                            activity_balances[coord_tech, activity.idx] * feedin_subject[coord_tech]) # Multiply the feed-in value by the element-wise product of activity balances and feedin_subject
                    feedin_effect_tech = np.maximum(feedin_effect_tech, 0) # Ensure nonnegative feedin effect

                    # Obtain the production costs per technology (nH x nT)
                    cost_prod = np.zeros((nH, nT)) # Preallocate
                    vom_cost_temp = vom_cost[coord_tech]  # length nT
                    energy_balances_temp = activity_balances[coord_tech][:, coord_energy]  # size: nT x (num energy coords)
                    emission_balances_temp = activity_balances[coord_tech][:, coord_emission]  # size: nT x (num emission coords)
                    for iT in range(nT):

                        # Identify activities with positive energy balances for this technology.
                        energy_pos = energy_balances_temp[iT, :] > 0  # boolean mask

                        # Identify cogeneration share (CHECK: computed but not used later)
                        if activity.energy_idx is not None:
                            denom = np.sum(energy_balances_temp[iT, energy_pos])
                            co_share = (energy_balances_temp[iT, activity.energy_idx] / denom) if denom != 0 else 0
                        else:
                            co_share = 1
                            emission_balances_temp[iT, activity.emission_idx] = 0 # Set the emission balance corresponding to this activity to zero.

                        # Modify the balances to include only consumed energy:
                        energy_balances_temp[iT, energy_pos] = 0

                        # Calculate production costs. Subtract variable costs, energy costs, emission costs, feedin subsidy and add taxes.
                        cost_prod[:, iT] = (
                            vom_cost_temp[iT]
                            - np.sum(energy_prices_hourly_out * (np.ones((nH, 1)) @ energy_balances_temp[iT, :].reshape(1, -1)), axis=1)
                            - np.sum(emission_prices_hourly * (np.ones((nH, 1)) @ emission_balances_temp[iT, :].reshape(1, -1)), axis=1)
                            - feedin_effect_tech[iT]
                            + taxes_effect
                        )

                    # Split the production costs per tech use share.
                    if activity.energy_idx is not None:
                        energy_prices_hourly_out[:, activity.energy_idx] = np.sum(cost_prod * share_prod, axis=1)
                    else:
                        emission_prices_hourly[:, activity.emission_idx] = np.sum(cost_prod * share_prod, axis=1)

                else:
                    if iI == 0 and activity.energy_idx is not None:
                        energy_prices_hourly_out[:, activity.energy_idx] = prices_hourly[:, activity.idx]

    # Obtain the yearly average prices
    energy_prices = np.zeros(nAe)
    for iAe in range(nAe):
        energy_prices[iAe] = np.sum(energy_prices_hourly_out[:, iAe]) / nH

    emission_prices = np.zeros(nAc)
    for iAc in range(nAc):
        emission_prices[iAc] = np.sum(emission_prices_hourly[:, iAc]) / nH

    # Adjust prices accordingly with the scarcity price
    # Obtain the price curve for all primary energy technologies that are not Electricity
    primary_available = np.zeros(nTb)
    primary_price = np.zeros(nTb)
    for tech in tech_entities:
        iTb = tech.idx
        act = tech.activity

        # Check if the technology is 'Primary' and its associated activity is not Electricity
        if (tech.category == 'Primary') and (act is not None) and (act.label != 'Electricity'):
            primary_available[iTb] = techStock[iTb] * cap2act[iTb] - tech_use[iTb]
            primary_price[iTb] = vom_cost[iTb]

    # Sort primary prices in ascending order and compute cumulative volume
    order = np.argsort(primary_price)
    primary_curve_price = primary_price[order]
    primary_curve_volume = np.cumsum(primary_available[order])

    # Obtain the scarcity price based on total scarcity
    total_scarcity = np.sum(energy_scarcity)
    scarcity_coord = min(int(np.sum(primary_curve_volume < total_scarcity)) + 1, len(primary_curve_price))
    scarcity_price = primary_curve_price[scarcity_coord - 1]

    # Check all energy activities for scarcity amd adjust their yearly prices
    for iAe, act in enumerate(energy_activities):

        # Check the temporal resolution of the dispatch:
        check_not_yearly = act.resolution in ['daily', 'hourly', 'hourly-interconnected']
        if not check_not_yearly:
            if energy_scarcity[iAe] > 0:
                old_price = energy_prices[iAe]
                energy_prices[iAe] = old_price + abs(old_price - scarcity_price) * scarcity_penalization
                print(f'.... Energy price of {act.name} was increased from {old_price:.2f} EUR/GJ to {energy_prices[iAe]:.2f} EUR/GJ due to scarcity.')

    # Obtain price ranges distribution for the hourly prices
    energy_prices_ranges = np.empty((nRp, nAe))
    for iAe in range(nAe):
        sorted_prices = np.sort(energy_prices_hourly_out[:, iAe])
        for iRp in range(nRp):
            nHr = price_ranges_hours[iRp]
            energy_prices_ranges[iRp, iAe] = np.mean(sorted_prices[:int(nHr)])

    # Save Variables
    activities.energies.prices.yearly[:, iP] = energy_prices
    activities.energies.prices.hourly[:, :, iP] = energy_prices_hourly_out
    activities.energies.prices.ranges[:, :, iP] = energy_prices_ranges
    activities.emissions.prices.yearly[:, iP] = emission_prices

    return activities
