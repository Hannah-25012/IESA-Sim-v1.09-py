from results_system_costs import results_system_costs
from results_emissions import results_emissions
from results_policy_cashflows import results_policy_cashflows

# File to process the new indicators for the results

def results_process(dimensions, types, parameters, activities, technologies, policies, results):
    # Obtain system costs
    print('--Obtaining system costs...')
    results = results_system_costs(dimensions, parameters, activities, technologies, results)

    # Obtain emissions indicators
    print('--Obtaining emission figures...')
    results = results_emissions(dimensions, types, activities, technologies, results)

    # Obtain policy cashflows
    print('--Obtaining policy cashflows...')
    results = results_policy_cashflows(dimensions, parameters, types, activities, technologies, policies, results)

    return results
