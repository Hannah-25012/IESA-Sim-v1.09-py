from results_process import results_process
from results_graph import results_graph
from results_write import results_write


# File to generate the results of the simulation
def mod5_results(dimensions, types, parameters, activities, technologies, agents, policies, results, outputName_root, plot_price_duration):
    # Process the results of the simulation into new indicators
    print("-Processing new indicators from the obtained variables... ")
    results = results_process(dimensions, types, parameters, activities, technologies, policies, results)

    # Call the graphing routine
    print("-Generating the graphs...")
    results_graph(dimensions, types, activities, technologies, results, plot_price_duration)

    # Call the writing routine
    print("-Generating the reports...")
    results_write(types, activities, technologies, agents, results, outputName_root)
    
    return results
