# File to dispatch interconnectors
import numpy as np

def disp_power_interconnectors(xc_efficiencies, xc_vom, xc_per_elec, elec_prices_hourly,
                                xc_demand, net_available_hourly):

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
#          5) xc_demand: float, matrix of each interconnector's own nominal
#             hourly capacity (nH,nI) - the most it could ever draw from its
#             "from" side that hour
#          6) net_available_hourly: float, matrix of each node's own local
#             generation capacity minus its own local demand (nH,nAk) -
#             positive is spare capacity, negative is a shortfall
#    Output:
#          1) xc_use_hourly: float, matrix of hourly use per interconnector
#             (nH,nI), as a fraction (0 to 1) of that interconnector's own
#             nominal capacity that hour

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

        # A positive spread only says trading is *profitable* that hour, not
        # how much to trade - dispatching the full nominal capacity every
        # such hour regardless of actual need was harmless while capacity
        # was small/broken, but once a link legitimately holds a large
        # capacity (e.g. after invest_techstocks_def.py's guaranteed-
        # availability treatment) it can report moving far more electricity
        # than the "from" side even generates. Bound the flow to the
        # smaller of: what the "to" node's own local generation actually
        # falls short by, and what the "from" node's own local generation
        # actually has spare - never more than either side's real physical
        # position, converted to a common ("from"-side draw) basis via the
        # interconnector's own efficiency.
        to_gap = np.maximum(-net_available_hourly[:, to_Ak], 0.0)
        from_surplus = np.maximum(net_available_hourly[:, from_Ak], 0.0)
        to_gap_as_draw = to_gap / xc_efficiencies[iI]
        capped_draw = np.minimum(to_gap_as_draw, from_surplus)
        capacity = xc_demand[:, [iI]]
        fraction = np.where(
            active_spread.astype(bool) & (capacity > 0),
            np.minimum(1.0, capped_draw / np.maximum(capacity, np.finfo(float).eps)),
            0.0,
        )

        # Save the flows
        xc_use_hourly[:, iI] = fraction.flatten()

    return xc_use_hourly