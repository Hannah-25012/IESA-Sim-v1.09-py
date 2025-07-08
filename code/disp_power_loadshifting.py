import numpy as np
import math

def disp_power_loadshifting(loadshifts_efficiencies, loadshifts_capacities, 
                             loadshifts_min, loadshifts_range, loadshifts_per_elec, 
                             loadshifts_demand_hourly, elec_prices_hourly, nDy, nHd):
    
# ========== Explanation ===============================================
# This function shifts technology demand from expensive to cheap hours in a
# time window. These shifts take into account innefficiencies and load
# shape constraints (min = load - 0, max = capacity - load)
#    Input:
#          1) loadshifts_efficiencies: float, vector of energy losses per
#             loadshift (nL)
#          2) loadshifts_capacities: float, vector of maximum value at  
#             which load can be shifted (nL)
#          3) loadshifts_min: float, vector of minimum load share for load  
#             to be shifted (nL)
#          4) loadshifts_range: float, vector of time windows for loadshift
#             to happen (nL)
#          5) loadshifts_per_elec: logical, matrix of loadshifting technologies
#             per electricity activity (nL,nAk)
#          6) loadshifts_demand_hourly: float, matrix of hourly demand of
#             the loadshift technologies(nH,nL)
#          7) elec_prices_hourly: float, matrix of hourly prices of the
#             electricity activities (nH,nAk)
#          8) nDy: integer, number of days per year
#          9) nHd: integer, number of hours per day
#    Output:
#          1) loadshifts_demand_hourly: float, matrix of hourly demand of
#             the loadshift technologies(nH,nL)

    # Extract dimensions
    nL = loadshifts_per_elec.shape[0]  # Number of generators
    nH = loadshifts_demand_hourly.shape[0]  # Number of hours

    # Obtain the hourly loadshifts of each technology
    loadshifts_hourly = np.zeros((nH, nL))  # Preallocate


    for iL in range(nL):
        
        # Obtain the number of periods windows for which shifting will occur
        nDl = loadshifts_range[iL]
        nW = math.ceil(nDy / nDl)

        # Identify the electricity activity
        iAk = loadshifts_per_elec[iL, :]

        # Apply the logic for each period
        for iW in range(nW):

            iH_start = iW * nDl * nHd
            
            if iW == nW -1:  # Final window
                iH_end = nH  # Include all remaining hours
            else:
                iH_end = iH_start + nDl * nHd  # Regular window size

            iH_window = np.arange(iH_start, iH_end, dtype=int)
            
            selected_cols = np.where(iAk == 1)[0]
            col_index = selected_cols[0]
            prices_vector = elec_prices_hourly[iH_window, col_index]

            demand = loadshifts_demand_hourly[iH_window, iL]
            room_down = demand - demand * loadshifts_min[iL]
            room_up = loadshifts_capacities[iL] - demand

            # Sort vectors by prices and identify the maximum transfer potential
            order = np.argsort(prices_vector, kind='stable')

            # Debugging
            # print("prices_vector[order]:")
            # print(prices_vector[order])

            order_up = np.cumsum(room_up[order])
            # Debugging
            # print("room_up(order):")
            # print(room_up[order])
            # print("order_up:")
            # print(order_up)

            # Longer, more manual way to match matlab "cumsum" precisely
            x = room_down[order[::-1]]   # same as flip(order) in MATLAB

            order_down = np.zeros_like(x)
            running_sum = 0.0

            # Summation from the end to the start:
            for i in reversed(range(len(x))):
                running_sum += x[i]
                order_down[i] = running_sum

            # Debugging
            # print("room_down(order):")
            # print(room_down[order])
            # print("order_down:")
            # print(order_down)
            # print("room_down[order[::-1]]")
            # print(room_down[order[::-1]])

            iSolution = np.argmin(np.abs(order_up - order_down))

            # Identify when the price gap is enough
            iBuy = iSolution
            iSell = iSolution + 1
            price_gap = -1
            dir_bin = 0

            # while iBuy > 0 and iSell < len(iH_window) and price_gap < 0:
            while iBuy > 0 and iSell < len(iH_window) -1 and price_gap < 0:
                dir_bin += 1
                if dir_bin % 2 == 1:
                    iBuy -= 1
                else:
                    iSell += 1

                price_buy = prices_vector[order[iBuy]]
                price_sell = prices_vector[order[iSell]]
                price_gap = price_sell - price_buy / loadshifts_efficiencies[iL]

            # If there is enough gap, activate the loadshifting
            if price_gap > 0:

                # Identify which volume to shift
                iH_buy = order[:iBuy + 1]
                iH_sell = order[iSell:]

                # precision = 14

                buy_volume = np.sum(room_up[iH_buy])
                sell_volume = np.sum(room_down[iH_sell])
                volume_shift = min(buy_volume, sell_volume)
                buy_adjust = volume_shift / buy_volume
                sell_adjust = volume_shift / sell_volume

                # Adjust the loadshift_hourly vector
                loadshifts_hourly[iH_buy, iL] = room_up[iH_buy] * buy_adjust
                loadshifts_hourly[iH_sell, iL] = -room_down[iH_sell] * sell_adjust

            # Debug
            # print(iW, iH_window[0], iH_window[-1])

    # Adjust the new demand
    loadshifts_demand_hourly += loadshifts_hourly

    #debug 
    # sum_loadshifts = np.sum(loadshifts_demand_hourly, axis=0)
    # print("Sum of loadshifts_demand_hourly columns:")
    # print(sum_loadshifts)


    return loadshifts_demand_hourly

   