# File to process the new indicators for the results
from results_system_costs import results_system_costs
from results_emissions import results_emissions
from results_policy_cashflows import results_policy_cashflows
from period_balances import replay_per_period

def results_process(dimensions, types, parameters, activities, technologies, policies, results):
    
    # Obtain system costs
    print('--Obtaining system costs...')
    results = results_system_costs(dimensions, parameters, activities, technologies, results)

    # Obtain emissions indicators
    print('--Obtaining emission figures...')
    results = replay_per_period(technologies, dimensions['nP'],
                                lambda: results_emissions(dimensions, types, activities, technologies, results),
                                results, ['emissions', 'emissions_sector_pos', 'emissions_sector_neg', 'emissions_stored'])

    # Obtain policy cashflows
    print('--Obtaining policy cashflows...')
    results = replay_per_period(technologies, dimensions['nP'],
                                lambda: results_policy_cashflows(dimensions, parameters, types, activities, technologies, policies, results),
                                results, ['policy_cashflows'])

    return results
