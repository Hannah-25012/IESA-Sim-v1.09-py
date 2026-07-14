# File to optimize (dispatch) batteries dispatch
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
#
# All nB batteries are solved together (the nH backward/forward DP passes
# are still one Python loop each - the recursion is inherently sequential
# across hours - but that loop no longer repeats per battery). Batteries
# have different charge-level state counts (bat_volume hours of storage),
# so every battery is padded to the largest state count and the states
# beyond its own range are firewalled off with -1e6 so they're never
# selected, instead of solving each battery's DP separately.

    # Extract dimensions
    nB = len(bat_efficiency)  # Number of battery technologies
    nAk = bat_per_elec.shape[1]
    nH = elec_prices_hourly.shape[0]  # Number of hours

    bat_use_hourly = np.zeros((nH, nB))
    bat_demand_elec_hourly = np.zeros((nH, nAk))

    if nB == 0:
        return bat_use_hourly, bat_demand_elec_hourly

    nSt_h_per_battery = np.round(np.asarray(bat_volume)).astype(int) + 1
    nSt_h_max = int(nSt_h_per_battery.max())
    nSt_a = 3  # activity dimension (discharging, nothing, charging)

    bat_stock_arr = np.asarray(bat_stock, dtype=float)
    charge_vec = -bat_stock_arr * np.asarray(bat_capacity) / np.asarray(bat_efficiency)
    discharge_vec = bat_stock_arr * np.asarray(bat_capacity)
    vom_vec = np.asarray(bat_vom)
    iAk_per_battery = np.array([np.where(bat_per_elec[iB, :])[0][0] for iB in range(nB)])

    # Which (state, action, battery) combinations are never allowed, regardless
    # of hour: states beyond a battery's own range (it's padding, to give every
    # battery the same array shape), charging while full, discharging while
    # empty. In the original per-battery (unpadded) array, the "while full"
    # boundary was always the array's last index, so slicing (`cont_values[...,
    # :-1, ...]`) naturally excluded it from ever being recomputed by the
    # recursion below. Once batteries are padded to a shared nSt_h_max, a
    # smaller battery's "full" state is no longer the last index, so the
    # recursion's `max()` update would silently overwrite it (and the padding)
    # with a real-looking value - this mask gets re-applied every hour to
    # keep those states firewalled at -1e6.
    invalid_mask = np.zeros((nSt_h_max, nSt_a, nB), dtype=bool)
    for iB in range(nB):
        nSt_h = nSt_h_per_battery[iB]
        if nSt_h < nSt_h_max:
            invalid_mask[nSt_h:, :, iB] = True
        invalid_mask[nSt_h - 1, 2, iB] = True  # avoid charging while full
    invalid_mask[0, 0, :] = True  # avoid discharging while empty

    # The matrix of continuation values: an hourly +1 dimension, a charging
    # level dimension, an activity dimension (discharging, nothing,
    # charging), and now a battery dimension.
    cont_values = np.zeros((nH + 1, nSt_h_max, nSt_a, nB))  # Preallocate
    cont_values[nH][invalid_mask] = -1e6
    cont_values[nH, 1:, :, :] = -1e6  # ensure every battery ends empty

    # Price calculation (each battery's own node's price, for every hour at once)
    price_hourly = elec_prices_hourly[:, iAk_per_battery]  # (nH, nB)
    charge_cashflow_hourly = price_hourly * charge_vec[None, :]
    discharge_cashflow_hourly = (price_hourly - min_spread - vom_vec[None, :]) * discharge_vec[None, :]

    for iH in range(nH - 1, -1, -1):

        # Update continuation values for each action state, for all batteries at once
        cont_values[iH, 1:, 0, :] = np.max(cont_values[iH + 1, :-1, :, :], axis=1) + discharge_cashflow_hourly[iH, :]  # Discharging
        cont_values[iH, :, 1, :] = np.max(cont_values[iH + 1, :, :, :], axis=1)  # Nothing
        cont_values[iH, :-1, 2, :] = np.max(cont_values[iH + 1, 1:, :, :], axis=1) + charge_cashflow_hourly[iH, :]  # Charging

        # Re-firewall states that must never be selected (see comment above)
        cont_values[iH][invalid_mask] = -1e6

    # Find the path of maximum values, for all batteries at once
    iSt_h = np.zeros(nB, dtype=int)  # Start empty
    battery_idx = np.arange(nB)
    for iH in range(nH):

        # Identify the optimal state per hour, per battery
        options_a = cont_values[iH, iSt_h, :, battery_idx]  # (nB, nSt_a)
        sel_opt = nSt_a - 1 - np.argmax(options_a[:, ::-1], axis=1)  # Match MATLAB 'last' occurrence

        # Define the state change
        delta_iSt_h = sel_opt - 1
        iSt_h = iSt_h + delta_iSt_h

        # Save results
        bat_use_hourly[iH, :] = bat_stock_arr * delta_iSt_h

        # Find the direction of change
        demand = np.where(delta_iSt_h == 1, -charge_vec, np.where(delta_iSt_h == -1, -discharge_vec, 0.0))
        np.add.at(bat_demand_elec_hourly[iH, :], iAk_per_battery, demand)

    return bat_use_hourly, bat_demand_elec_hourly
