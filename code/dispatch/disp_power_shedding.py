# File to dispatch shedding technologies
import numpy as np

def disp_power_shedding(shed_guarantee, shed_maxvolume_hourly,
                        shed_minvolume_hourly, shed_per_elec, elec_prices_hourly,
                        shed_maxdemand_hourly=None, shed_mindemand_hourly=None,
                        spare_hourly=None):

# ========== Explanation ===============================================
# This function presents an approach to calculate when will shedding
# technologies will operate. Based on the known prices vector, the sedding
# technologies choose the cheapest hours required to meet their guaranties.
#    Input:
#          1) shed_guarantee: float, vector of required guarantees per
#             shedding technology (nS,1)
#          2) shed_maxVolume_hourly: float, matrix of hourly max Volumes
#             per shedding technology (nH,nS)
#          3) shed_minVolume_hourly: float, matrix of hourly min Volumes
#             per shedding technology (nH,nS)
#          4) shed_per_elec: logical, matrix of shedding technologies per
#             electricity activity (nS,nAk)
#          5) elec_prices_hourly: float, matrix of hourly prices of the
#             electricity activities (nH,nAk)
#          6) shed_maxdemand_hourly/shed_mindemand_hourly (nH,nS) and
#             spare_hourly (nH,nAk), all optional and only used together:
#             the node's real spare generation per hour AFTER every
#             shedding technology's minimum demand is reserved. When given,
#             a technology only runs (consumes its flexible band) in an hour
#             up to the spare that hour still has, and its use becomes a
#             fraction in [0,1] instead of on/off.
#    Output:
#          1) shed_use_hourly: float, matrix of hourly use per shedding
#             technology (nH,nS)
#
# Capacity-aware mode mirrors IESA-Opt, where a shedding technology is a
# bounded term INSIDE the hourly node balance (deltaS_shed_TS, never allowed
# to break supply = demand) rather than a free choice of hours: without the
# spare bound, every flexible consumer picks the same cheapest hours, piles
# its whole band on top of each other, and pushes those hours into shortage
# (VOLL) - then reacts to the resulting price spike next iteration, cycling.

    # Extract dimensions
    nS = len(shed_guarantee)  # Number of shedding technologies
    nH = elec_prices_hourly.shape[0]  # Number of hours

    capacity_aware = (shed_maxdemand_hourly is not None and shed_mindemand_hourly is not None
                      and spare_hourly is not None)
    if capacity_aware:
        spare_left = np.array(spare_hourly, dtype=float, copy=True)

    # For each tecnology, obtain the most profitable hours to operate in a year
    shed_use_hourly = np.zeros((nH, nS))
    for iS in range(nS):

        # Identify the prices vector of the technology
        cols = np.where(shed_per_elec[iS, :].astype(bool))[0]
        if len(cols) == 1:
            prices_vector = elec_prices_hourly[:, cols[0]]
            prices_order   = np.argsort(prices_vector, kind='stable')
        else:
            pv_flat = elec_prices_hourly[:, cols].flatten(order='F')
            flat_idx = np.argsort(pv_flat, kind='stable')
            prices_order = flat_idx % nH

        # Get the normalized flexible volume
        flex_volume = shed_maxvolume_hourly[:, iS] - shed_minvolume_hourly[:, iS]

        # Get the flexible target
        flex_target = shed_guarantee[iS] - shed_minvolume_hourly[:, iS].sum()

        if not capacity_aware:
            # Identify which hours satisfy the guarantee constraint
            cumsum_volume = np.cumsum(flex_volume[prices_order])
            last_on = np.searchsorted(cumsum_volume, flex_target, side='left')
            shed_use_hourly[prices_order[:last_on + 1], iS] = 1
            continue

        # Capacity-aware fill: cheapest hours first, but never more flexible
        # demand in an hour than the node has spare left there (the node is
        # the technology's own when it has one, else the first it touches).
        node = cols[0]
        flex_demand = shed_maxdemand_hourly[:, iS] - shed_mindemand_hourly[:, iS]
        remaining = flex_target
        seen = set()
        for h in prices_order:
            if remaining <= 0:
                break
            if h in seen:
                continue
            seen.add(h)
            fv, fd = flex_volume[h], flex_demand[h]
            if fv <= 0 or fd <= 0:
                continue
            frac = min(1.0, remaining / fv, max(spare_left[h, node], 0.0) / fd)
            if frac <= 0:
                continue
            shed_use_hourly[h, iS] = frac
            spare_left[h, node] -= frac * fd
            remaining -= frac * fv

    return shed_use_hourly
