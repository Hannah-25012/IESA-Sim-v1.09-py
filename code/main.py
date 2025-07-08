import time
import os
import pickle
from pathlib import Path
from mod0_read_data import mod0_read_data
from mod1_initialize import mod1_initialize
from mod2_invest import mod2_invest
from mod3_dispatch import mod3_dispatch
from mod4_postprocessing import mod4_postprocessing
from mod5_results import mod5_results

def main(settings):
    # Record start time
    t_start = time.perf_counter()

    # Beginning message
    print("We are now going to start solving the defined simulation scenario")

    # Implement the settings
    input_file = settings['input']
    output_path = os.path.join('output', settings['scenario_name'])
    read_input = settings['read_input']
    save_output = settings['save_output']
    year_end = settings['year_end']
    plot_price_duration = settings['plot_price_duration']
    print("Settings were successfully implemented")

    # Read the data
    if read_input is True:
        print(f"Reading the excel file {input_file} ...")
        parameters, types, activities, profiles, technologies, agents, policies = mod0_read_data(input_file)
        print("The input data was retrieved successfully from the excel file")
    else:
        # The following part of code loads data from .mat file (if we want to ensure compatibility with matlab)
        # from scipy.io import loadmat
        # data = loadmat('input/data.mat')
        # parameters = data['parameters']
        # types = data['types']
        # activities = data['activities']
        # profiles = data['profiles']
        # technologies = data['technologies']
        # agents = data['agents']
        # policies = data['policies']
        # print("The input data was successfully loaded from .mat file")


        with open('data.pkl', 'rb') as file:
            data = pickle.load(file)
        parameters = data['parameters']
        types = data['types']
        activities = data['activities']
        profiles = data['profiles']
        technologies = data['technologies']
        agents = data['agents']
        policies = data['policies']
        print("The input data was successfully loaded from .pkl file")

    print(f"Time elapsed: {time.perf_counter() - t_start:.2f} seconds")

    # Initialize the simulation
    print("Initializing the simulation...")
    dimensions, types, activities, technologies, results = mod1_initialize(
        settings, types, activities, technologies, agents, policies
    )
    print("Initialization complete.")
    print(f"Time elapsed: {time.perf_counter() - t_start:.2f} seconds")

    # Begin the sequential solution on all periods
    print("Simulation is on...")
    periods = activities['periods']
    n_periods = sum(year_end >= periods)

    # Loop through the periods
    for iP in range(n_periods):
        print(f"Now solving the period: {periods[iP]}...")

        # Call the invest module
        print(f"-Determining investments for the period: {periods[iP]}...")
        activities, technologies = mod2_invest(
            dimensions, parameters, activities, technologies, agents, policies, iP
        )
        print("-Investment module routine complete.")
        print(f"Time elapsed: {time.perf_counter() - t_start:.2f} seconds")

        # Call the energy module
        print(f"-Determining the operation of technologies for the period: {periods[iP]}...")
        activities, technologies = mod3_dispatch(
            dimensions, parameters, activities, technologies, profiles, policies, iP
        )
        print("-Operation of technologies determined.")

        # Call the post-process module
        print(f"-Postprocessing parameters for the period: {periods[iP]}...")
        activities, technologies, results = mod4_postprocessing(
            dimensions, parameters, types, activities, technologies, profiles, policies, results, iP
        )
        print("-Parameters postprocessed.")

        print(f"Period {periods[iP]} was finalized.")
        print(f"Time elapsed: {time.perf_counter() - t_start:.2f} seconds")

    print("Simulation is over.")

    # Generate the results
    print("Generating the results post-simulation...")
    results = mod5_results(
        dimensions, types, parameters, activities, technologies, agents, policies, results, output_path, plot_price_duration
    )
    print("Results generated successfully.")

    # Save output to .mat file (if interaction with matlab is desired)
    # if save_output:
    #     output_file = os.path.join(output_path, '_variables.mat')
    #     from scipy.io import savemat
    #     savemat(output_file, {
    #         'dimensions': dimensions,
    #         'parameters': parameters,
    #         'types': types,
    #         'activities': activities,
    #         'technologies': technologies,
    #         'profiles': profiles,
    #         'results': results
    #     })
    #     print(f"Output saved to {output_file}")


    # Save output to .pkl file (better if the code runs only in Python)
    if save_output:
        output_file = Path(output_path) / 'simulation_results.pkl'
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'wb') as file:
            pickle.dump({
                'dimensions': dimensions,
                'parameters': parameters,
                'types': types,
                'activities': activities,
                'technologies': technologies,
                'profiles': profiles,
                'results': results
            }, file)
        print(f"Output successfully saved to {output_file}")


    # Print the last time stamp
    print(f"Total time elapsed: {time.perf_counter() - t_start:.2f} seconds")
