# File to solve the dispatch of daily activities
import numpy as np

def disp_gas(dimensions, parameters, activities, technologies, profiles, policies,
             tech_use_hourly, prices_hourly, iP):

    # Extract Parameters
    nH  = dimensions['nH']
    nHd = dimensions['nHd']
    nDy = dimensions['nDy']
    nTb = dimensions['nTb']
    gas_premium = parameters.scarcity.gas_premium
    periods = activities.periods
    activities_names = activities.names
    gaseous_activities = [a for a in activities.entities if a.is_gaseous]
    dispatchType_tech = technologies.balancers.dispatch
    activity_balances = technologies.balancers.activity_balances
    tech_stock = technologies.balancers.stocks.evolution[:, iP]
    cap2act = technologies.balancers.cap2acts
    vom_cost = technologies.balancers.costs.voms[:, iP]
    hourly_profile_tech = technologies.balancers.profiles
    shedding_capacity = technologies.balancers.shedding.capacity
    buffer_up = technologies.balancers.buffers.up
    buffer_down = technologies.balancers.buffers.down
    buffer_capacity = technologies.balancers.buffers.capacity
    profileType = profiles.types
    hourly_profiles = profiles.shapes
    # Each hourly profile type's shape, resolved once instead of re-scanning
    # profileType for every technology that uses it.
    profile_shape_by_type = {name: hourly_profiles[:, i] for i, name in enumerate(profileType)}
    feedin_subject = technologies.balancers.policies.feedin_subject
    feedin_activities = policies.feedins.activities
    feedin_values = policies.feedins.values
    taxes_activities = policies.taxes.activities
    taxes_values = policies.taxes.values
    taxes_idx_by_name = {name: i for i, name in enumerate(taxes_activities)}
    feedin_idx_by_name = {name: i for i, name in enumerate(feedin_activities)}

    # First we solve the yearly balance of operations
    # Dispatch technologies must meet the demand for all operation technologies
    coord_sheddingAll = shedding_capacity > 0
    for activity in gaseous_activities:

        # Identify the activity coord (the single column matching this activity)
        coord_act = np.zeros(len(activities_names), dtype=bool)
        coord_act[activity.idx] = True

        # Identify the main activity technologies (activity.technologies, its children)
        coord_main = np.zeros(nTb, dtype=bool)
        coord_main[[t.idx for t in activity.technologies]] = True
        coord_tech = coord_main & (np.array(dispatchType_tech) != 'Gas buffer')
        coord_demand = ~coord_main

        # Identify the dispatch type 'Daily dispatch'
        coord_dispatch = coord_tech & (np.array(dispatchType_tech) == 'Daily dispatch')
        coord_operate = coord_tech & (~coord_dispatch) & (~coord_sheddingAll)
        coord_shedding = coord_tech & (coord_sheddingAll)

        # First non-dispatchable technologies operate at their times
        nTo = np.sum(coord_operate)
        techUse_iTo = tech_stock[coord_operate] * cap2act[coord_operate]
        profiles_use = [p for p, flag in zip(hourly_profile_tech, coord_operate) if flag]
        techUse_hourly_iTo = np.zeros((nH, int(nTo)))
        for iTo in range(int(nTo)):
            techUse_hourly_iTo[:, iTo] = techUse_iTo[iTo] * profile_shape_by_type[profiles_use[iTo]]
        tech_use_hourly[:, coord_operate] = techUse_hourly_iTo
        tech_use = np.sum(tech_use_hourly, axis=0)

        # Identify the yearly resulting demand for the activity
        act_demand = - np.sum( tech_use[coord_demand].reshape(-1, 1) *
                                activity_balances[coord_demand, :][:, coord_act] )

        # Identify the share of the demand already met by operating technologies
        generation_operation = np.sum( tech_use[coord_operate].reshape(-1, 1) *
                                       activity_balances[coord_operate, :][:, coord_act] )
        generation_shedding = np.sum( tech_use[coord_shedding].reshape(-1, 1) *
                                      activity_balances[coord_shedding, :][:, coord_act] )
        residual_demand = act_demand - generation_operation - generation_shedding
        weighted_stock = (tech_stock[coord_dispatch] * cap2act[coord_dispatch])[:, np.newaxis] # Calculate the elementwise product of the two 1D arrays and reshape to (n,1)
        selected_activity = activity_balances[np.ix_(coord_dispatch, coord_act)] # Select the submatrix of activity_balances using np.ix_
        dispatch_potential = np.sum(weighted_stock * selected_activity) # Multiply elementwise and sum all elements

        # Display warning message if there's not enough installed capacity
        if residual_demand > dispatch_potential:
            print(f"!!! Warning: there is not enough installed dispatchable capacity to meet the residual demand of {activity.name} in the year {periods[iP]}")
            print(f".... Residual demand is {residual_demand:6.2f} PJ and max supply is {dispatch_potential:6.2f} PJ")

        # If there is residual demand < 0 then we are generating an excess
        if residual_demand < 0:
            print(f"!!! Warning: there is an excess production of {activity.name} in the year {periods[iP]}")
            print(f".... Residual demand is {residual_demand:6.2f} PJ")
            utilization_rate = 0
        else:

            # Utilization share of the dispatchable technologies
            if dispatch_potential > 0:
                utilization_rate = min(1, residual_demand / dispatch_potential)
            else:
                utilization_rate = 0

        # Use of dispatch technologies
        nTd = np.sum(coord_dispatch)
        techUse_iTd = utilization_rate * tech_stock[coord_dispatch] * cap2act[coord_dispatch]
        profiles_use = [p for p, flag in zip(hourly_profile_tech, coord_dispatch) if flag]
        techUse_hourly_iTd = np.zeros((nH, int(nTd)))
        for iTd in range(int(nTd)):
            techUse_hourly_iTd[:, iTd] = techUse_iTd[iTd] * profile_shape_by_type[profiles_use[iTd]]
        tech_use_hourly[:, coord_dispatch] = techUse_hourly_iTd
        tech_use = np.sum(tech_use_hourly, axis=0)

        # Identify the price value
        coord_noact = ~coord_act

        # Select the technologies that will define the prices
        coord_prices = coord_tech  # use coord_tech for all techs or coord_dispatch for the dispatch ones only
        if np.sum(tech_use[coord_prices]) > 0:

            # We identify if there is a tax for this activity
            taxes_effect = 0
            if activity.name in taxes_idx_by_name:
                taxes_effect = taxes_values[taxes_idx_by_name[activity.name], iP]

            # We identify if there is a feed-in for this activity
            feedin_effect_tech = 0
            if activity.name in feedin_idx_by_name:
                feedin_effect_tech = feedin_values[feedin_idx_by_name[activity.name], iP] * (
                    activity_balances[coord_prices, activity.idx] *
                    feedin_subject[coord_prices]
                )
            feedin_effect_tech = np.maximum(feedin_effect_tech, 0)

            # Obtain the price accordingly with the average of production costs
            term1 = np.sum( tech_use[coord_prices] *
                           (vom_cost[coord_prices] - feedin_effect_tech) ) # weighted yearly average numerator (vom cost adjusted for feedin)
            A = tech_use_hourly[:, coord_prices]
            B = prices_hourly[:, coord_noact]
            C = activity_balances[coord_prices, :][:, coord_noact]
            term2 = np.sum( A * np.dot(B, C.T) ) # weighted yearly average fuel cost component
            act_price = (term1 - term2) / np.sum(tech_use[coord_prices])

            # We adjust for taxes
            act_price = act_price + taxes_effect

            # We cap to positive values
            act_price = max(act_price, 0)

            # Report the hourly price (which will be changed below)
            prices_hourly[:, coord_act] = act_price

    # Now we solve the gas dispatch at a daily level where we identify the
    # largest storage peak needed, and the largest storage volume needed
    for activity in gaseous_activities: # CHECK: for the first gaseous activity (for Iid = 1), the tech_use of the buffer technologies 466 and 467 is modified and is incorrect. this is because hourly_balance is off from before.

        # Identify the coordinate
        coord_act = np.zeros(len(activities_names), dtype=bool)
        coord_act[activity.idx] = True
        coord_main = np.zeros(nTb, dtype=bool)
        coord_main[[t.idx for t in activity.technologies]] = True
        coord_buffer = coord_main & (np.array(dispatchType_tech) == 'Gas buffer')

        # Obtain the hourly/daily balance of the activity
        hourly_balance = np.sum(tech_use_hourly * (np.ones((nH, 1)) @ activity_balances[:, coord_act].T), axis=1)

        if np.any(hourly_balance != 0):
            daily_balance = np.zeros((nDy, 1)) # Preallocate
            for iDy in range(nDy):

                # Identify which hours
                coord_hours = np.arange(iDy * nHd, (iDy + 1) * nHd) # (MATLAB: (iDy-1)*nHd + 1 : iDy*nHd)

                # Define the daily balance
                daily_balance[iDy, 0] = np.sum(hourly_balance[coord_hours])

            # Cummulative volumes per day
            daily_cummulative = np.cumsum(daily_balance, axis=0)

            # Buffer requirements
            buffer_upward_peak = np.max(daily_balance)
            buffer_downward_peak = np.min(daily_balance)
            buffer_days = np.max(daily_cummulative) / buffer_downward_peak

            # Buffer shares
            buffers_up = buffer_up[coord_buffer]
            buffers_down = buffer_down[coord_buffer]
            buffers_capacities = buffer_capacity[coord_buffer]
            buffers_shares = 1 - np.abs(buffers_capacities / buffer_days - 1)
            buffers_shares = buffers_shares / np.sum(buffers_shares)

            # Buffer installations
            buffers_installations_up = buffer_upward_peak / np.sum(buffers_up * buffers_shares)
            buffers_installations_down = buffer_downward_peak / np.sum(buffers_down * buffers_shares)
            if buffers_installations_up > buffers_installations_down:
                buffers_stock = buffer_upward_peak * buffers_shares
            else:
                buffers_stock = buffer_downward_peak * buffers_shares

            # Save the buffers data
            tech_stock[coord_buffer] = buffers_stock
            tech_use_hourly[:, coord_buffer] = - np.outer(hourly_balance, buffers_stock)

            # Modify the hourly price shapes
            avg_price = np.mean(prices_hourly[:, coord_act])
            max_val = np.max(-hourly_balance)
            min_val = np.min(-hourly_balance)
            adjustment_vector = 1 - gas_premium + 2 * gas_premium * ((-hourly_balance - min_val) / (max_val - min_val))

            prices_hourly[:, coord_act] = prices_hourly[:, coord_act] * adjustment_vector.reshape(-1, 1)

            # Ensure average values in the intended levels
            prices_hourly[:, coord_act] = prices_hourly[:, coord_act] * avg_price / np.mean(prices_hourly[:, coord_act])

    return tech_use_hourly, prices_hourly, tech_stock
