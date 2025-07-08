import numpy as np

from disp_initialize_power import disp_initialize_power
from disp_power_generators import disp_power_generators
from disp_power_shedding import disp_power_shedding
from disp_power_loadshifting import disp_power_loadshifting
from disp_power_batteries import disp_power_batteries
from disp_power_interconnectors import disp_power_interconnectors

def disp_power(dimensions, parameters, activities, technologies, profiles, tech_use_hourly, prices_hourly, iP, nId):
    # Extract parameters
    nDy = dimensions['nDy']
    nHd = dimensions['nHd']
    nH = dimensions['nH']
    nIp = dimensions['nIp']
    nAk = dimensions['nAk']
    voll = parameters['voll']
    min_spread = parameters['min_spread']
    activities_label = activities['labels']
    activities_elec = activities['electricity']['names']

    # Initialize power dispatch
    if nId:
        print('---Initializing power dispatch...')

    (
        gen_vom, gen_balance_hourly, gen_availability_hourly,
        gen_xc_costs_hourly, gen_per_elec, elec_demand_hourly,
        shed_guarantee, shed_maxvolume_hourly, shed_minvolume_hourly, shed_per_elec,
        shed_maxdemand_hourly, shed_mindemand_hourly, shed_multiplier,
        loadshifts_efficiencies, loadshifts_capacities, loadshifts_min, loadshifts_per_uoa,
        loadshifts_range, loadshifts_per_elec, loadshifts_demand_hourly,
        bat_efficiency, bat_capacity, bat_volume, bat_per_elec, bat_vom, bat_stock,
        xc_efficiencies, xc_vom, xc_per_elec, xc_demand, xc_supply, elec_cogenerated,
        tech_generators_coord, tech_shedding_coord, tech_loadshifts_coord,
        tech_batteries_coord, tech_interconnectors_coord
    ) = disp_initialize_power(dimensions, activities, technologies, profiles, tech_use_hourly, iP)

    # Display header of the price table
    if nId:
        print('---Beginning of the power dispatch iterations...')
        print('==================== Average electricity price per node in EUR/MWh ====================')
        print('======================= Average electricity load per node in TWh ======================')
        print(f"{'Iter':6}, {'Type':10}", end='')
        for iAk in range(nAk):
            print(f"{activities_elec[iAk]:24}", end='')
        print()

    # Begin the iterative loop
    elec_demand_hourly_new = elec_demand_hourly.copy()
    for iIp in range(nIp):
        # Dispatch generators
        gen_use_hourly, elec_prices_hourly = disp_power_generators(
            gen_vom, gen_balance_hourly, gen_availability_hourly,
            gen_xc_costs_hourly, gen_per_elec, elec_demand_hourly_new,
            prices_hourly, voll
        )

        # Display iteration info
        if nId:
            print(f"{iIp:6d} {'Price':10}", end='')
            for iAk in range(nAk):
                print(f"{np.mean(elec_prices_hourly[:, iAk]) * 3.6:24.2f}", end='')
            print()
            print(f"{iIp:6d} {'Load':10}", end='')
            for iAk in range(nAk):
                print(f"{(sum(elec_demand_hourly_new[:, iAk]) + elec_cogenerated[iAk, 0]) / 3.6:24.2f}", end='')
            print()

        # Dispatch shedding technologies
        if sum(tech_shedding_coord):
            shed_use_hourly = disp_power_shedding(
                shed_guarantee, shed_maxvolume_hourly,
                shed_minvolume_hourly, shed_per_elec, elec_prices_hourly
            )

        # Dispatch load-shifting technologies
        if sum(tech_loadshifts_coord) > 0:
            loadshifts_demand_hourly = disp_power_loadshifting(
                loadshifts_efficiencies, loadshifts_capacities,
                loadshifts_min, loadshifts_range, loadshifts_per_elec,
                loadshifts_demand_hourly, elec_prices_hourly, nDy, nHd
            )

        # Dispatch batteries
        if sum(tech_batteries_coord) > 0:
            bat_use_hourly, bat_demand_elec_hourly = disp_power_batteries(
                bat_efficiency, bat_capacity, bat_volume,
                bat_vom, bat_stock, bat_per_elec, elec_prices_hourly, min_spread
            )

        # Dispatch interconnectors
        xc_use_hourly = disp_power_interconnectors(
            xc_efficiencies, xc_vom, xc_per_elec, elec_prices_hourly
        )

        # Adjust demand for each electricity activity
        for iAk in range(nAk):
            # Identify technologies in the node
            is_shedding = shed_per_elec[:, iAk]
            is_loadshifts = loadshifts_per_elec[:, iAk]
            is_interconnect_from = np.sum(xc_per_elec[:, iAk, :], axis=0).T[:, None].astype(bool)
            is_interconnect_to = np.sum(xc_per_elec[iAk, :, :], axis=0).T[:, None].astype(bool)



            # Add base demand
            elec_demand_hourly_new[:, iAk] = elec_demand_hourly[:, iAk]

            # Add shedding technology demand
            if np.sum(tech_shedding_coord) > 0:  # Check if there are any active shedding technologies
                elec_demand_hourly_new[:, iAk] += np.sum(
                    shed_mindemand_hourly[:, is_shedding] +
                    shed_use_hourly[:, is_shedding] * (
                        shed_maxdemand_hourly[:, is_shedding] - shed_mindemand_hourly[:, is_shedding]
                    ),
                    axis=1  # Sum along rows (equivalent to MATLAB's sum(..., 2))
                )

            # Add load-shifting demand
            if sum(tech_loadshifts_coord) > 0:
                elec_demand_hourly_new[:, iAk] += loadshifts_demand_hourly[:, is_loadshifts].sum(axis=1)

            # Add battery demand
            if sum(tech_batteries_coord) > 0:
                elec_demand_hourly_new[:, iAk] += bat_demand_elec_hourly[:, iAk]

            # Add interconnector demand and supply
            elec_demand_hourly_new[:, iAk] += (
                xc_demand[:, is_interconnect_from.ravel()] * xc_use_hourly[:, is_interconnect_from.ravel()]
            ).sum(axis=1) + (
                xc_supply[:, is_interconnect_to.ravel()] * xc_use_hourly[:, is_interconnect_to.ravel()]
            ).sum(axis=1)

    # Display message
    if nId:
        print('---End of the power dispatch iterations.')

    # Save the variables
    activities_elec_coord = [label == 'Electricity' for label in activities_label]
    prices_hourly[:, activities_elec_coord] = elec_prices_hourly

    tech_use_hourly[:, tech_generators_coord] = gen_use_hourly

    if sum(tech_shedding_coord):
        tech_use_hourly[:, tech_shedding_coord] = (
            shed_mindemand_hourly +
            shed_use_hourly * (
                shed_maxdemand_hourly - shed_mindemand_hourly
            )
        ) / (np.ones((nH, 1)) @ shed_multiplier.reshape(1, -1))

    if sum(tech_loadshifts_coord) > 0:
        tech_use_hourly[:, tech_loadshifts_coord] = (
            loadshifts_demand_hourly / (
                np.ones((nH, 1)) @ loadshifts_per_uoa.reshape(1, -1) + np.finfo(float).eps
            )
        )

    if sum(tech_batteries_coord) > 0:
        tech_use_hourly[:, tech_batteries_coord] = bat_use_hourly

    tech_use_hourly[:, tech_interconnectors_coord] = xc_demand * xc_use_hourly

    # debugging 
    # Sum along the rows (axis=0)
    # tech_use_sum = np.sum(tech_use_hourly, axis=0)
    # prices_sum = np.sum(prices_hourly, axis=0)

    # Print the results
    # print("Sum of tech_use_hourly across rows:", tech_use_sum)
    # print("Sum of prices_hourly across rows:", prices_sum)

    return tech_use_hourly, prices_hourly
