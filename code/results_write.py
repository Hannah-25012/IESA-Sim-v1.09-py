import pandas as pd

from write_system_costs import write_system_costs
from write_system_emisisons import write_system_emisisons
from write_sectoral_emissions import write_sectoral_emissions
from write_policy_cashflows import write_policy_cashflows
from write_techUse import write_techUse
from write_techStock import write_techStock
from write_investments import write_investments
from write_sector_balance import write_sector_balance
from write_energy_prices import write_energy_prices
from write_emission_prices import write_emission_prices
from write_LCOPs import write_LCOPs
from write_MCAs import write_MCAs
from write_choices import write_choices
from write_hourly_power_prices import write_hourly_power_prices
from write_hourly_gas_prices import write_hourly_gas_prices


# File to generate the output reports
def results_write(types, activities, technologies, agents, results, output_name_root):
    # Write the general results file
    print('--Writting the general output file...')
    general_output_name = output_name_root + '_general.xlsx'

    with pd.ExcelWriter(general_output_name, engine='openpyxl') as writer:

        # Write System Costs
        write_system_costs(activities, results, writer)
        # Write System Emissions
        write_system_emisisons(activities, results, writer)
        # Write Sectoral Emissions
        write_sectoral_emissions(activities, types, results, writer)
        # Write Policy Cashflows
        write_policy_cashflows(types, activities, results, writer)
        # Write techUse sheet
        write_techUse(activities, technologies, writer)
        # Write techStock sheet
        write_techStock(activities, technologies, writer)
        # Write investments sheet
        write_investments(activities, technologies, writer)
        # Write the sectoral balances sheet
        write_sector_balance(types, activities, technologies, writer)
        # Write the energy prices
        write_energy_prices(activities, writer)
        # Write the emission prices
        write_emission_prices(activities, writer)
        # Write the LCOPs page
        write_LCOPs(activities, technologies, writer)
        # Write the MCAs page
        write_MCAs(activities, technologies, agents, writer)
        # Write the agent choices page
        write_choices(activities, technologies, agents, writer)

    # Write the hourly output file
    print('--Writting the hourly output file...')
    hourly_output_name = output_name_root + '_power_prices_hourly.xlsx'
    write_hourly_power_prices(activities, hourly_output_name)
    hourly_output_name = output_name_root + '_gas_prices_hourly.xlsx'
    write_hourly_gas_prices(activities, hourly_output_name)