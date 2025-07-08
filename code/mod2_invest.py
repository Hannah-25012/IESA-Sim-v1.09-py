import numpy as np
from invest_retrofit_potential import invest_retrofit_potential
from invest_investment_potential import invest_investment_potential
from invest_drivers_sufficiency import invest_drivers_sufficiency
from invest_tech_LCOPs import invest_tech_LCOPs
from invest_multicriteria_performance import invest_multicriteria_performance
from invest_tech_choices_per_act import invest_tech_choices_per_act
from invest_driver_technologies import invest_driver_technologies
from invest_techstock_check import invest_techstock_check
from invest_nerghg_technologies import invest_nerghg_technologies
from invest_ccus_technologies import invest_ccus_technologies
from invest_energy_technologies import invest_energy_technologies
from invest_power_technologies import invest_power_technologies
from invest_techstocks_def import invest_techstocks_def
from invest_decommissioning import invest_decommissioning


def mod2_invest(dimensions, parameters, activities, technologies, agents, policies, iP):
    
    # Attention: Double-check Operations like subtraction, addition, and dot products are correct and align with Matlab
    # Extract parameters
    if iP > 0:
        tech_stock_exist = technologies['balancers']['stocks']['evolution'][:, iP - 1]
    else:
        tech_stock_exist = technologies['balancers']['stocks']['initial']
    decommissionings = technologies['balancers']['decommissionings'][:, iP]

    # Decommission the technologies in the period
    print("--Updating the existing stocks after each period decommissioning...")
    tech_stock_new_original = tech_stock_exist - decommissionings
    # Attention: Determine whether .copy() that seperates tech_stock-new from tech_stock_original is needed. The variable is used again later in invest_techstocks_def
    tech_stock_new = tech_stock_new_original.copy()

    # Determine the retrofitting potentials
    print("--Determining retrofitting potentials...")
    retrofit_sources, retrofit_options, retrofit_potential, retrofit_cost = invest_retrofit_potential(
        dimensions, technologies, tech_stock_new
    )

    # Determine investment potentials
    print("--Determining investment potentials...")
    investment_potential = invest_investment_potential(
        dimensions, technologies, tech_stock_new, retrofit_potential, iP
    )

    # Determine the driver gaps and invest
    print("--Determining the gaps to meet driver activities volumes...")
    activity_gap = invest_drivers_sufficiency(
        dimensions, activities, technologies, tech_stock_new, False, iP
    )

    # Obtain the technology LCOPs of the period
    print("--Obtaining LCOPs of technologies...")
    technologies = invest_tech_LCOPs(
        dimensions, activities, technologies, policies, retrofit_potential, retrofit_cost, iP
    )

    # Call the multi criteria performance function
    print("--Measuring multi criteria performance...")
    technologies = invest_multicriteria_performance(
        dimensions, activities, technologies, agents, iP
    )

    # Call the technology choices function
    print("--Determining technology choices per agent populations...")
    technologies, tech_choices = invest_tech_choices_per_act(
        dimensions, activities, technologies, agents, iP
    )

    # Determine the new stocks, the old stocks
    print("--Determining the required investments to meet driver activities volumes...")
    tech_stock_new, investments_drivers = invest_driver_technologies(
        dimensions, activities, technologies, tech_stock_new, investment_potential, activity_gap, tech_choices, iP
    )

    # Check if there are negative investments
    if (investments_drivers < 0).any():
        print("!!!Negative investments within drivers routine!!!")
        input("Press Enter to continue...")

    # Determine the sufficiency of the new driver stocks
    print("--Revising that tech stocks are respecting constraints for driver activities...")
    tech_stock_new = invest_techstock_check(
        dimensions, activities, technologies, tech_stock_new, tech_stock_exist, False, iP
    )

    # Invest in nonER-GHG emission technologies
    print("--Determining the required investments to mitigate non-energy related GHG...")
    tech_stock_new, investments_nerghg = invest_nerghg_technologies(
        dimensions, activities, technologies, tech_stock_new, investment_potential, tech_choices, iP
    )

    # Invest in CCUS technologies
    # Attention: Different value for ccus_gap for activity number 127! Something must be wrong, not sure what...
    # this is probably because of rounding. Need to adjust rounding in module in matlab and python and check again.
    print("--Determining the required investments to mitigate non-energy related GHG...")
    tech_stock_new, investments_ccus = invest_ccus_technologies(
        dimensions, activities, technologies, tech_stock_new, investment_potential, tech_choices, iP
    )

    # Determine the energy investments
    print("--Determining the required investments to satisfy energy activities...")
    technologies, activities, tech_stock_new, investments_energy = invest_energy_technologies(
        dimensions, activities, technologies, tech_stock_new, investment_potential, tech_choices, iP
    )

    # Check if there are negative investments
    if (investments_energy < 0).any():
        print("!!!Negative investments within energy routine!!!")
        input("Press Enter to continue...")

    # Invest in power activities but not for the first period
    # Attention: module invest_power_technologies is only called from the second period onwards, so still needs to be debugged!
    if iP > 0:
        print("--Investing in power generation technologies...")
        investments_power = invest_power_technologies(
            dimensions, parameters, activities, technologies, agents, tech_stock_new, iP
        )
    else:
        investments_power = np.zeros_like(investments_energy)

    # Check if there are negative investments
    if (investments_power < 0).any():
        print("!!!Negative investments within power routine!!!")
        input("Press Enter to continue...")

    # Preliminary investments
    print("--Defining final investments...")
    preliminary_investments = (
        investments_drivers + investments_nerghg + investments_ccus + investments_energy + investments_power
    )

    # Determine the definitive tech stocks that will be used for the operation of the system
    print("--Determining the definitive tech Stocks...")
    technologies, forced_decommissionings = invest_techstocks_def(
        dimensions, technologies, tech_stock_new_original, preliminary_investments, False, iP
    )

    # Calculate the decommissionings due to investments
    print("--Determining the decommissionings based on the investments...")
    technologies = invest_decommissioning(
        dimensions, activities, technologies, forced_decommissionings, retrofit_sources,
        retrofit_options, retrofit_potential, tech_stock_exist, iP
    )

    return activities, technologies