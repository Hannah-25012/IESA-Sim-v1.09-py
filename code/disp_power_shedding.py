import numpy as np

def disp_power_shedding(shed_guarantee, shed_maxvolume_hourly, 
                        shed_minvolume_hourly, shed_per_elec, elec_prices_hourly):
   
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
#    Output: 
#          1) shed_use_hourly: float, matrix of hourly use per shedding
#             technology (nH,nS)


    # Extract dimensions
    nS = len(shed_guarantee)  # Number of shedding technologies
    nH = elec_prices_hourly.shape[0]  # Number of hours

    # Initialize the result matrix
    shed_use_hourly = np.zeros((nH, nS))

    # Loop through each shedding technology
    for iS in range(nS):

        # Identify the prices vector of the technology
        prices_vector = elec_prices_hourly[:, shed_per_elec[iS, :].astype(bool)]

        # Get the normalized flexible volume
        flex_volume = shed_maxvolume_hourly[:, iS] - shed_minvolume_hourly[:, iS]

        # Get the flexible target
        flex_target = shed_guarantee[iS] - np.sum(shed_minvolume_hourly[:, iS])

        # Order the hours of preference
        prices_order = np.argsort(prices_vector, axis=0).flatten()

        # Identify which hours satisfy the guarantee constraint
        cumsum_volume = np.cumsum(flex_volume[prices_order])
        last_on = np.searchsorted(cumsum_volume, flex_target, side='left')
        shed_use_hourly[prices_order[:last_on + 1], iS] = 1

    return shed_use_hourly