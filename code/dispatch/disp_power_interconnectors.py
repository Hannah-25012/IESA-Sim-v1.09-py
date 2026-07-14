# File to dispatch interconnectors
import numpy as np

def disp_power_interconnectors(xc_efficiencies, xc_vom, xc_per_elec, elec_prices_hourly):
    
# ========== Explanation ===============================================
# This function allocates flows of electricity per interconnector based on
# positive gradients between nodal prices (adjusted for loses and costs)
#    Input:
#          1) xc_efficiencies: float, vector of energy losses per
#             interconnector (nI)
#          2) xc_vom: float, vector of variable transport costs per
#             interconnector (nI)
#          3) xc_per_elec: logical, cube with matrixes to x from per
#             interconnector (nAk,nAk,nI)
#          4) elec_prices_hourly: float, matrix of hourly prices of the
#             electricity activities (nH,nAk)
#    Output:
#          1) xc_use_hourly: float, matrix of hourly use per interconnector
#             (nH,nI)

    # Extract dimensions
    nI = len(xc_efficiencies)  # Number of interconnectors
    nH = elec_prices_hourly.shape[0]  # Number of hours
    
    # For each hour, obtain the functional spreads per interconnector
    xc_use_hourly = np.zeros((nH, nI))
    for iI in range(nI):

        # Identify from and to networks
        from_Ak, to_Ak = np.unravel_index(
            np.flatnonzero(xc_per_elec[:, :, iI]), xc_per_elec[:, :, iI].shape, order='F')

        # Prices in the areas
        to_prices = elec_prices_hourly[:, to_Ak]
        from_prices = elec_prices_hourly[:, from_Ak]
        
        # Obtain spreads
        spreads = np.maximum(to_prices - (from_prices / xc_efficiencies[iI] - xc_vom[iI]), 0)

        # NOTE: These commented out lines are present in matlab, but will_vec and abs_vec are anyway overwritten by the alternative approach.
        # will_min = np.percentile(spreads, 17)
        # will_max = np.percentile(spreads, 66)
        # abs_min = np.percentile(from_prices, 17)
        # abs_max = np.percentile(from_prices, 66)

        # Active spread linearly increasing from will_min to will_max
        # will_vec = np.minimum(np.maximum((spreads - will_min) / (will_max - will_min), 0), 1)
        
        # From prices linearly increasing from abs_max to abs_min
        # if abs_min == np.mean(from_prices) and abs_max == np.mean(from_prices):
        #     abs_vec = 1
        # else:
        #     abs_vec = np.minimum(np.maximum((from_prices - abs_min) / (abs_max - abs_min + np.finfo(float).eps), 0), 1)
        
        # Manuel says "try other approach"
        will_vec = spreads > 0
        abs_vec = 1   
        active_spread = will_vec * abs_vec
        
        # Save the flows
        xc_use_hourly[:, iI] = active_spread.flatten()

    return xc_use_hourly