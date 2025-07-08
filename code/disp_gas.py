import numpy as np

def disp_gas(dimensions, parameters, activities, technologies, profiles, policies,
             tech_use_hourly, prices_hourly, iP):
    
    # Extract Parameters
    nH   = dimensions['nH']
    nHd  = dimensions['nHd']
    nDy  = dimensions['nDy']
    nAg  = dimensions['nAg']
    gas_premium = parameters['scarcity']['gas_premium']
    periods = activities['periods']
    activities_names = activities['names']
    activities_gaseous = activities['gaseous']['names']
    activityPer_tech = technologies['balancers']['activities']
    activity_balances = technologies['balancers']['activity_balances']
    tech_stock = technologies['balancers']['stocks']['evolution'][:, iP]
    dispatchType_tech = technologies['balancers']['dispatch']
    cap2act = technologies['balancers']['cap2acts']
    vom_cost = technologies['balancers']['costs']['voms'][:, iP]
    hourly_profile_tech = technologies['balancers']['profiles']
    shedding_capacity = technologies['balancers']['shedding']['capacity']
    buffer_up = technologies['balancers']['buffers']['up']
    buffer_down = technologies['balancers']['buffers']['down']
    buffer_capacity = technologies['balancers']['buffers']['capacity']
    profileType = profiles['types']
    hourly_profiles = profiles['shapes']
    feedin_subject = technologies['balancers']['policies']['feedin_subject']
    feedin_activities = policies['feedins']['activities']
    feedin_values = policies['feedins']['values']
    taxes_activities = policies['taxes']['activities']
    taxes_values = policies['taxes']['values']

    
    # First we solve the yearly balance of operations
    # Dispatch technologies must meet the demand for all operation technologies
    coord_sheddingAll = shedding_capacity > 0

    for iAg in range(nAg):
        # Identify the activity coord
        coord_act = np.array([name == activities_gaseous[iAg] for name in activities_names])
        
        # Identify the main activity technologies
        coord_tech = (np.array(activityPer_tech) == activities_gaseous[iAg]) & \
                     (np.array(dispatchType_tech) != 'Gas buffer')
        coord_demand = (np.array(activityPer_tech) != activities_gaseous[iAg])
        
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
            coord_profile = np.array([pt == profiles_use[iTo] for pt in profileType])
            col_index = np.where(coord_profile)[0]
            if col_index.size == 0:
                raise ValueError("No matching profile found for non-dispatchable technology")
            col_index = col_index[0]
            techUse_hourly_iTo[:, iTo] = techUse_iTo[iTo] * hourly_profiles[:, col_index]
        tech_use_hourly[:, coord_operate] = techUse_hourly_iTo
        tech_use = np.sum(tech_use_hourly, axis=0)
        
        # Identify the yearly resulting demand for the activity
        act_demand = - np.sum( tech_use[coord_demand].reshape(-1, 1) *
                                activity_balances[coord_demand, :][:, coord_act] )
        
        generation_operation = np.sum( tech_use[coord_operate].reshape(-1, 1) *
                                       activity_balances[coord_operate, :][:, coord_act] )
        generation_shedding = np.sum( tech_use[coord_shedding].reshape(-1, 1) *
                                      activity_balances[coord_shedding, :][:, coord_act] )
        residual_demand = act_demand - generation_operation - generation_shedding

        # Calculate the elementwise product of the two 1D arrays and reshape to (n,1)
        weighted_stock = (tech_stock[coord_dispatch] * cap2act[coord_dispatch])[:, np.newaxis]

        # Select the submatrix of activity_balances using np.ix_
        selected_activity = activity_balances[np.ix_(coord_dispatch, coord_act)]

        # Multiply elementwise and sum all elements
        dispatch_potential = np.sum(weighted_stock * selected_activity)

        
        # Display warning message if there's not enough installed capacity
        if residual_demand > dispatch_potential:
            print(f"!!! Warning: there is not enough installed dispatchable capacity to meet the residual demand of {activities_gaseous[iAg]} in the year {periods[iP]}")
            print(f".... Residual demand is {residual_demand:6.2f} PJ and max supply is {dispatch_potential:6.2f} PJ")
        
        # If there is residual demand < 0 then we are generating an excess
        if residual_demand < 0:
            print(f"!!! Warning: there is an excess production of {activities_gaseous[iAg]} in the year {periods[iP]}")
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
            coord_profile = np.array([pt == profiles_use[iTd] for pt in profileType])
            col_index = np.where(coord_profile)[0]
            if col_index.size == 0:
                raise ValueError("No matching profile found for dispatch technology")
            col_index = col_index[0]
            techUse_hourly_iTd[:, iTd] = techUse_iTd[iTd] * hourly_profiles[:, col_index]
        tech_use_hourly[:, coord_dispatch] = techUse_hourly_iTd
        tech_use = np.sum(tech_use_hourly, axis=0)
        
        # Identify the price value
        coord_noact = ~coord_act
        
        # Select the technologies that will define the prices
        coord_prices = coord_tech  # use coord_tech for all techs or coord_dispatch for the dispatch ones only
        if np.sum(tech_use[coord_prices]) > 0:
            # We identify if there is a tax for this activity
            act_for_tax = np.array(activities_names)[coord_act]
            coord_taxes_act = np.array([a in taxes_activities for a in act_for_tax])
            taxes_effect = 0
            if np.sum(coord_taxes_act) > 0:
                # Assuming only one tax applies; take the first matching index
                matching_indices = [i for i, a in enumerate(act_for_tax) if a in taxes_activities]
                if matching_indices:
                    tax_index = taxes_activities.index(act_for_tax[matching_indices[0]])
                    taxes_effect = taxes_values[tax_index, iP]
            
            # We identify if there is a feed-in for this activity
            act_for_feedin = np.array(activities_names)[coord_act]
            coord_feedin_act = np.array([a in feedin_activities for a in act_for_feedin])
            feedin_effect_tech = 0
            if np.sum(coord_feedin_act) > 0:
                matching_indices = [i for i, a in enumerate(act_for_feedin) if a in feedin_activities]
                if matching_indices:
                    feedin_index = feedin_activities.index(act_for_feedin[matching_indices[0]])
                    feedin_effect_tech = feedin_values[feedin_index, iP] * (
                        activity_balances[coord_prices, :][:, coord_act] *
                        feedin_subject[coord_prices]
                    )
            feedin_effect_tech = np.maximum(feedin_effect_tech, 0)
            
            # Obtain the price accordingly with the average of production costs
            # First term: weighted yearly average numerator (vom cost adjusted for feedin)
            term1 = np.sum( tech_use[coord_prices] *
                           (vom_cost[coord_prices] - feedin_effect_tech) )
            # Second term: weighted yearly average fuel cost component
            A = tech_use_hourly[:, coord_prices]
            B = prices_hourly[:, coord_noact]
            C = activity_balances[coord_prices, :][:, coord_noact]
            term2 = np.sum( A * np.dot(B, C.T) )
            act_price = (term1 - term2) / np.sum(tech_use[coord_prices])
            
            # We adjust for taxes
            act_price = act_price + taxes_effect
            
            # We cap to positive values
            act_price = max(act_price, 0)
            
            # Report the hourly price (which will be changed below)
            prices_hourly[:, coord_act] = act_price

    
    # Now we solve the gas dispatch at a daily level where we identify the
    # largest storage peak needed, and the largest storage volume needed


    for iAg in range(nAg):
    # Attention: In iAg = 0 (for Iid = 1), the tech_use of the buffer technologies 466 and 467 is modified and is incorrect.
    # this is because hourly_balance is off from before.
    # Identify the coordinate
        coord_act = np.array([name == activities_gaseous[iAg] for name in activities_names])
        # debugging
        print("coord_act sum:", np.sum(coord_act))  # how many columns are True?
        print("coord_act where True:", np.where(coord_act)[0]) # which activity is selected?

        coord_buffer = (np.array(activityPer_tech) == activities_gaseous[iAg]) & (np.array(dispatchType_tech) == 'Gas buffer')
        # debugging
        print("coord_buffer sum:", np.sum(coord_buffer))
        print("coords in coord_buffer:", np.where(coord_buffer)[0])

        print("tech_use_hourly sum", np.sum(tech_use_hourly))


        # debugging
        # print("activities_gaseous[iAg]:", repr(activities_gaseous[iAg]))
        # for name in activities_names:
        #     if name == activities_gaseous[iAg]:
        #         print("Matched:", repr(name))
        # print("coord_act sum = ", np.sum(coord_act))

        # debugging - here, tech_use_hourly(coord_buffer) is still correct! Further down it's modified and incorrect.
        print("sum of tech_use_hourly coord buffer (1)", np.sum(tech_use_hourly[:, coord_buffer], axis=0))
        
        # Obtain the hourly/daily balance of the activity
        # Attention: hourly_balance doesn't match, values incorrect in iId = 1, iAg = 0.
        # values for activity balance[:. coord_act] are correct.
        # So there must be an issue with tech_use_hourly
        hourly_balance = np.sum(tech_use_hourly * (np.ones((nH, 1)) @ activity_balances[:, coord_act].T), axis=1)

        # debugging
        # print("sum of activity balances coord act", np.sum(activity_balances[:, coord_act]))

        # debugging
        # print("tech_use_hourly:", tech_use_hourly.shape)
        # print("activity_balances:", activity_balances.shape)
        
        # print("activity_balances[:, coord_act].shape:", activity_balances[:, coord_act].shape)
        balance_mult = np.ones((nH,1)) @ activity_balances[:, coord_act].T
        print("balance_mult shape:", balance_mult.shape)
        print("balance_mult sum:", np.sum(balance_mult)) # this is correct

        hourly_balance_mult = tech_use_hourly * balance_mult

        tol = 1e-20
        hourly_balance_mult[np.abs(hourly_balance_mult) < tol] = 0.0 # to eliminate -0 in my python code - but this doesn't help

        hourly_balance_2 = np.sum(hourly_balance_mult, axis=1)

        print("sum of hourly balance with intermediate steps", np.sum(hourly_balance_2))

        # print("Any NaN in the partial sums?", np.isnan(hourly_balance_2).any()) # false
        # print("Any Inf in the partial sums?", np.isinf(hourly_balance_2).any()) # false

        print("Min / max tech_use_hourly:", np.nanmin(tech_use_hourly), np.nanmax(tech_use_hourly)) # correct
        print("Min / max balance_mult:", np.nanmin(balance_mult), np.nanmax(balance_mult)) # correct

        # debugging (Note: hourly_balance_sum = some values are incorrect, activity_balances_sum = correct)
        hourly_balance_sum = np.sum(hourly_balance)
        print ("sum of hourly_balance:", hourly_balance_sum)
        # activity_balances_sum = np.sum(activity_balances, axis=0)
        # print("sum of activity_balances per activity:", activity_balances_sum)
        # print("sum of sum of activity_balances:", np.sum(activity_balances_sum)) # this is fine!
        # still debugging:
        # for j in range(550):
        #     w = activity_balances[j, coord_act]  # shape (1,) but effectively a scalar
        #     if w != 0:
        #         print(f"Tech j={j}, multiplier={w}, sum tech_use_hourly={tech_use_hourly[:, j].sum()}")
        # tolerance = 1e-14
        # for j in range(550):
        #     w = activity_balances[j, coord_act]
        #     if abs(w) > tolerance:  # catch very small but nonzero
        #         print(f"Tech j={j}, multiplier={w}, sum tech_use_hourly={tech_use_hourly[:, j].sum()}")


        
        if np.any(hourly_balance != 0):
            daily_balance = np.zeros((nDy, 1))
            for iDy in range(nDy):
                # Identify which hours (MATLAB: (iDy-1)*nHd + 1 : iDy*nHd)
                coord_hours = np.arange(iDy * nHd, (iDy + 1) * nHd)
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
            # debugging
            print("sum of tech_use_hourly coord buffer (2)", np.sum(tech_use_hourly[:, coord_buffer], axis=0))

            # print("hourly_balance shape:", hourly_balance.shape)
            # print("buffers_stock shape:", buffers_stock.shape)
            # print("coord_buffer sum:", np.sum(coord_buffer))
            # print("outer shape:", np.outer(hourly_balance, buffers_stock).shape)

            # tech_use_hourly_sum = np.sum(tech_use_hourly, axis=0)
            # print("sum of tech_use_hourly per technology:", tech_use_hourly_sum)
            
            # Modify the hourly price shapes
            avg_price = np.mean(prices_hourly[:, coord_act])
            max_val = np.max(-hourly_balance)
            min_val = np.min(-hourly_balance)
            adjustment_vector = 1 - gas_premium + 2 * gas_premium * ((-hourly_balance - min_val) / (max_val - min_val))
            
            prices_hourly[:, coord_act] = prices_hourly[:, coord_act] * adjustment_vector.reshape(-1, 1)
            
            # Ensure average values in the intended levels
            prices_hourly[:, coord_act] = prices_hourly[:, coord_act] * avg_price / np.mean(prices_hourly[:, coord_act])

    # debugging (Note: values in iId = 1 don't match, technologies 466 and 467 are gas buffer.)
    # tech_use_hourly_sum = np.sum(tech_use_hourly, axis=0)
    # print("sum of tech_use_hourly per technology:", tech_use_hourly_sum)
    
    # debugging 
    # techStock_sum = np.sum(techStock)
    # Sum along the rows (axis=0)
    # tech_use_sum = np.sum(techUse_hourly, axis=0)
    # prices_sum = np.sum(prices_hourly, axis=0)

    # Print the results
    # print ("Sum of techStock:", techStock_sum)
    # print("Sum of tech_use_hourly across rows:", tech_use_sum)
    # print("Sum of prices_hourly across rows:", prices_sum)


    return tech_use_hourly, prices_hourly, tech_stock
