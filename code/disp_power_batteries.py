import numpy as np

def disp_power_batteries(bat_efficiency, bat_capacity, bat_volume,
                        bat_vom, bat_stock, bat_per_elec, elec_prices_hourly, min_spread):
    
#  ========== Explanation ===============================================
# This function presents a DP optimization approach to determine operation
# of batteries per hournode taking into account known prices.
# Interconnectors, shedding, and generators are dispatched in other files.
#    Input:
#          1) bat_efficiency: float, vector of battery efficiencies (nB,1)
#          2) bat_capacity: float, vector of battery capacities (nB,1)
#          3) bat_volume: float, vector of battery volumes in hours (nB,1)
#          4) bat_vom: float, vector of battery vom costs (nB,1)
#          5) bat_stock: float, installed capacity of the battery (nB,1)
#          6) bat_per_elec: logical, matrix of batteries installed per
#             electricity activity (nB,nAk)
#          7) elec_prices_hourly: float, matrix of hourly prices of the
#             electricity activities (nH,nAk)
#    Output:
#          1) bat_use_hourly: float, matrix of hourly use per battery
#             (nH,nB)
#          2) bat_demand_elec_hourly: float, matrix of hourly electricity demand
#             per battery (nH,nAk)


    # Extract dimensions
    nB = len(bat_efficiency)  # Number of battery technologies
    nAk = bat_per_elec.shape[1]
    nH = elec_prices_hourly.shape[0]  # Number of hours
    
    # Preallocate result arrays
    bat_use_hourly = np.zeros((nH, nB))
    bat_demand_elec_hourly = np.zeros((nH, nAk))
    bat_demand = np.zeros((nAk, 1))
    bat_supply = np.zeros((nAk, 1))
    
    for iB in range(nB):
        # Define the number of states
        nSt_h = round(bat_volume[iB]) + 1
        nSt_a = 3  # activity dimension (discharging, nothing, charging)
        
        # Obtain battery parameters
        charge = -bat_stock[iB] * bat_capacity[iB] / bat_efficiency[iB]
        discharge = bat_stock[iB] * bat_capacity[iB]
        vom = bat_vom[iB]
        iAk = bat_per_elec[iB, :].astype(bool)
        
        # Initialize continuation values matrix
        cont_values = np.zeros((nH + 1, nSt_h, nSt_a))
        cont_values[nH, 1:nSt_h, :] = -1e6  # Ensure battery ends empty
        cont_values[:, 0, 0] = -1e6  # Avoid discharging while empty
        cont_values[:, nSt_h - 1, 2] = -1e6  # Avoid charging while full
        
        for iH in range(nH - 1, -1, -1):
            # Price calculation
            price = elec_prices_hourly[iH, iAk]
            charge_cashflow = price * charge
            discharge_cashflow = (price - min_spread - vom) * discharge
            
            # Update continuation values
            cont_values[iH, 1:, 0] = np.max(cont_values[iH + 1, :-1, :], axis=1) + discharge_cashflow  # Discharging
            cont_values[iH, :, 1] = np.max(cont_values[iH + 1, :, :], axis=1)  # Nothing
            cont_values[iH, :-1, 2] = np.max(cont_values[iH + 1, 1:, :], axis=1) + charge_cashflow  # Charging
        
        # Find the path of maximum values
        iSt_a_vec = np.zeros((nH, 1), dtype=int)
        iSt_h_vec = np.zeros((nH, 1), dtype=int)
        iSt_h = 0  # Start empty
        
        for iH in range(nH):
            # Identify the optimal state per hour
            options_a = cont_values[iH, iSt_h, :]
            sel_opt = len(options_a) - 1 - np.argmax(options_a[::-1])  # Match MATLAB 'last' occurrence
            
            # Define the state change
            delta_iSt_h = sel_opt - 1
            iSt_h = iSt_h + delta_iSt_h
            
            # Save decision
            iSt_a_vec[iH] = sel_opt
            iSt_h_vec[iH] = iSt_h
            
            # Find the direction of change
            if delta_iSt_h == 1:
                demand = -charge
                bat_demand[iAk] += demand
            elif delta_iSt_h == -1:
                demand = -discharge
                bat_supply[iAk] += demand
            else:
                demand = 0
            
            # Save results
            bat_use_hourly[iH, iB] = bat_stock[iB] * delta_iSt_h
            bat_demand_elec_hourly[iH, iAk] += demand

    #debug 
    # sum_bat_use = np.sum(bat_use_hourly, axis=0)
    # print("Sum of bat_use_hourly columns:")
    # print(sum_bat_use)

    #debug 
    # sum_bat_demand_elec = np.sum(bat_demand_elec_hourly, axis=0)
    # print("Sum of bat_demand_elec_hourly columns:")
    # print(sum_bat_demand_elec)
    
    return bat_use_hourly, bat_demand_elec_hourly