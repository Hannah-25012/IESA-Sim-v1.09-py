# Function to determine the operation of the energy assets for the period.

import numpy as np
from disp_energy_balance import disp_energy_balance
from disp_energy_intrayearly import disp_energy_intrayearly
from disp_energy_scarcity import disp_energy_scarcity
from disp_infrastructure import disp_infrastructure

def mod3_dispatch(dimensions, parameters, activities, technologies, profiles, policies, iP):

    # Extract Parameters
    nId = dimensions['nId']
    nTb = dimensions['nTb']
    nH = dimensions['nH']
    initialized_prices = activities['prices']['initialized'][:, iP]

    # Initialize prices
    prices_hourly = np.ones((nH, 1)) * initialized_prices

    # Call the iteration loop
    tech_use_hourly = np.zeros((nH, nTb))  # Preallocate (in units of activity)

    for iId in range(nId):

        # Call the energy balance for yearly activities
        print(f"--Solving the energy balance for the iteration: {iId + 1}...")
        _, tech_use_hourly = disp_energy_balance(
            dimensions, activities, technologies, profiles, tech_use_hourly, False, iP
        )
        # This is the first time already where there is a slight difference for tech_use_hourly 
        # debugging
        # tech_use_hourly_sum = np.sum(tech_use_hourly, axis=0)
        # print("sum of sum of tech_use_hourly:", np.sum(tech_use_hourly_sum))

        # Call the intrayearly energy balance module
        print("--Solving the intrayear energy balance in the iteration loop...")
        tech_use_hourly, prices_hourly, tech_stock = disp_energy_intrayearly(
            dimensions, parameters, activities, technologies, profiles, policies, 
            tech_use_hourly, prices_hourly, iP, iId == nId - 1
        )
        # debugging
        # tech_use_hourly_sum = np.sum(tech_use_hourly, axis=0)
        # print("sum of tech_use_hourly per technology:", tech_use_hourly_sum)
        # print("sum of sum of tech_use_hourly:", np.sum(tech_use_hourly_sum))

    # Call the energy balance to rebalance the dispatch
    print("--Solving the definitive energy balance to close the iteration loop ...")
    tech_use, tech_use_hourly = disp_energy_balance(
        dimensions, activities, technologies, profiles, tech_use_hourly, True, iP
    )

    # Call the energy scarcity quantifying module
    print("--Quantifying the energy scarcity in the period ...")
    activities = disp_energy_scarcity(dimensions, activities, technologies, iP)

    # Call the infrastructure module
    print("--Calculating the infrastructure needs ...")
    tech_stock_infra, investments_infra, tech_stock, investments = disp_infrastructure(
        dimensions, activities, technologies, tech_use_hourly, tech_stock, iP
    )

    # Save Variables
    activities['prices']['hourly'][:, :, iP] = prices_hourly # slightly off in last few decimals
    technologies['balancers']['use']['yearly'][:, iP] = tech_use # slightly off in last few decimals. matlab values 467 amd 468 don't match (python 466 and 467)
    technologies['balancers']['use']['hourly'][:, :, iP] = tech_use_hourly # slightly off in last few decimals
    # debugging
    # matlab values 467 amd 468 don't match (python 466 and 467). This concerns small scale storage buffer - Final Gas, large scale storage buffer - Final gas
    # tech_use_hourly_sum = np.sum(tech_use_hourly, axis=0)
    # print("sum of tech_use_hourly per technology:", tech_use_hourly_sum)
    technologies['balancers']['stocks']['evolution'][:, iP] = tech_stock # correct
    technologies['balancers']['investments'][:, iP] = investments # correct
    technologies['infra']['stocks']['evolution'][:, iP] = tech_stock_infra # correct
    technologies['infra']['investments'][:, iP] = investments_infra # correct

    return activities, technologies