import numpy as np

def disp_power_generators(gen_vom, gen_balance_hourly, gen_availability_hourly,
                          gen_xc_costs_hourly, gen_per_elec, elec_demand_hourly,
                          prices_hourly, voll):
    
# ========== Explanation ===============================================
# This function presents a merit order approach to determine prices per
# node taking into account only generation and load per hour.
# Interconnectors, shedding, and batteries are dispatched in other files.
#    Input:
#          1) gen_marginal_costs_hourly: float, matrix of houry marginal
#             costs per generator (nH,nG)
#          2) gen_availability_hourly: float, matrix of hourly
#             availabilities per generator (nH,nG)
#          3) elec_demand_hourly: float, matrix of hourly demand per
#             electricity activity (nH,nAk)
#          4) gen_per_elec: logical, matrix of generators installed per
#             electricity activity (nG,nAk)
#    Output:
#          1) gen_use_hourly: float, matrix of hourly use per generator
#             (nH,nG)
#          2) elec_prices_hourly: float, matrix of hourly prices of the
#             electricity activities (nH,nAk)


    # Extract dimensions
    nG = gen_per_elec.shape[0]  # Number of generators
    nAk = gen_per_elec.shape[1]  # Number of electricity activities
    nH = gen_xc_costs_hourly.shape[0]  # Number of hours

    # debugging 
    # correct
    # gen_vom_sum = np.sum(gen_vom)
    # Sum along the rows (axis=0)
    # incorrect
    # gen_balance_hourly_sum = np.sum(gen_balance_hourly, axis=0)
    # correct
    # prices_hourly_sum = np.sum(prices_hourly, axis=0)
    # correct
    # gen_xc_costs_hourly_sum = np.sum(gen_xc_costs_hourly, axis=0)

    # Print the results
    # print ("Sum of gen_vom:", gen_vom_sum)
    # print("Sum of gen_balance_hourly across rows:", gen_balance_hourly_sum)
    # print("Sum of prices_hourly across rows:", prices_hourly_sum)
    # print("Sum of gen_xc_costs_hourly across rows:", gen_xc_costs_hourly_sum)

    # Obtain the hourly marginal costs of the generator
    # Attention: wrong numbers in second iteration!
    gen_marginal_costs_hourly = np.zeros((nH, nG))  # Preallocate
    for iG in range(nG):
        gen_marginal_costs_hourly[:, iG] = (
            gen_vom[iG]  # variable cost component
            - np.sum(gen_balance_hourly[:, :, iG] * prices_hourly, axis=1)  # fuel cost component
            + gen_xc_costs_hourly[:, iG]  # interconnector hourly cost component
        )

    # For each hour, obtain the merit order curve, the marginal generator, the
    # resulting dispatch, and the marginal price per electricity activity
    gen_use_hourly = np.zeros((nH, nG))
    elec_prices_hourly = np.zeros((nH, nAk))

    for iH in range(nH):
        for iAk in range(nAk):

            # Define the problem to solve
            # iG = gen_per_elec[:, iAk].astype(bool)
            mask_gens = gen_per_elec[:, iAk].astype(bool)
            # Note: demand incorrect for iAk = 7 --> need to check elec_demand_hourly 
            demand = elec_demand_hourly[iH, iAk]

            gen_available = np.zeros(nG)
            gen_cost = np.zeros(nG)
            gen_available[mask_gens] = gen_availability_hourly[iH, mask_gens]
            gen_cost[mask_gens] = gen_marginal_costs_hourly[iH, mask_gens]


            # Obtain merit order
            MOC_order = np.argsort(gen_cost, kind='stable')
            MOC_volume = np.cumsum(gen_available[MOC_order])

            # Obtain marginal generator, volume, and price
            MOC_last_gen_indices = np.where(MOC_volume >= demand)[0]
            if MOC_last_gen_indices.size == 0:
                # Not enough volume in total => use all + VOLL price
                voll_true = True
                MOC_last_gen = nG - 1  # last generator index in 0-based
                MOC_excess = 0
            else:
                voll_true = False
                MOC_last_gen = MOC_last_gen_indices[0]
                MOC_excess = MOC_volume[MOC_last_gen] - demand
            
            
            iG_online = MOC_order[: MOC_last_gen + 1]
            iG_marginal = MOC_order[MOC_last_gen]

            if voll_true:
                elec_prices_hourly[iH, iAk] = voll
            else:
                elec_prices_hourly[iH, iAk] = gen_cost[iG_marginal]

            # Fill dispatch: set all online to full availability, then subtract excess from marginal
            gen_use_hourly[iH, iG_online] = gen_available[iG_online]
            gen_use_hourly[iH, iG_marginal] -= MOC_excess

    # debug
    # sum_gen_use = np.sum(gen_use_hourly, axis=0)
    # print("Sum of gen_use_hourly columns:")
    # print(sum_gen_use)
    #debug 
    # sum_elec_prices = np.sum(elec_prices_hourly, axis=0)
    # print("Sum of elec_prices_hourly columns:")
    # print(sum_elec_prices)

    return gen_use_hourly, elec_prices_hourly
