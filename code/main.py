# Main file from which all modules are called sequentially
import os
import sys

# The pipeline is organized by phase into subfolders (read/, initialize/, invest/,
# dispatch/, postprocess/, write/), but every module still uses flat
# `from module import thing` imports rather than package-qualified ones, so each
# phase folder needs to be on sys.path directly.
_code_dir = os.path.dirname(os.path.abspath(__file__))
for _phase_dir in ('read', 'initialize', 'invest', 'dispatch', 'postprocess', 'write'):
    _phase_path = os.path.join(_code_dir, _phase_dir)
    if _phase_path not in sys.path:
        sys.path.insert(0, _phase_path)

import time
import pickle
from pathlib import Path
from mod0_read_data_save_duck import mod0_read_data_save_duck
from mod0_load_duckdb import load_from_duckdb
from mod1_initialize import mod1_initialize
from entities import link_entities
from mod2_invest import mod2_invest
from mod3_dispatch import mod3_dispatch
from mod4_postprocessing import mod4_postprocessing
from mod5_results import mod5_results
from mod6_save_duckdb import save_state_duckdb, save_excel_duckdb

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

    # Read the data from the excel input file into the database, if requested
    db_path = "SIMmodel.duckdb"
    if read_input is True:
        print(f"Reading the excel file {input_file} into {db_path} ...")
        mod0_read_data_save_duck(input_file)
        print("The input data was retrieved successfully from the excel file")

    # Either way, the simulation runs off the database, not off dicts built
    # straight from Excel or off a stale pickle - it's the single source of
    # truth from here on.
    print(f"Loading input data from {db_path} ...")
    parameters, types, activities, profiles, technologies, agents, policies = load_from_duckdb(db_path)
    print("The input data was successfully loaded from the database")

    print(f"Time elapsed: {time.perf_counter() - t_start:.2f} seconds")

    # Initialize the simulation
    print("Initializing the simulation...")
    dimensions, types, activities, technologies, results = mod1_initialize(
        settings, types, activities, technologies, agents, policies
    )
    print("Initialization complete.")

    # Build the Activity/Technology/Infrastructure object graph (real
    # cross-references, resolved once here) on top of the now-finalized
    # arrays, so invest/dispatch/post-processing code can navigate
    # relationships (tech.activity, activity.technologies, ...) instead of
    # re-deriving index lists.
    activity_entities, tech_entities, infra_entities = link_entities(activities, technologies)
    activities.entities = activity_entities
    technologies.balancers.entities = tech_entities
    technologies.infra.entities = infra_entities
    print(f"Time elapsed: {time.perf_counter() - t_start:.2f} seconds")

    # Begin the sequential running of modules for all periods
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

        # Message for finalized period
        print(f"Period {periods[iP]} was finalized.")
        print(f"Time elapsed: {time.perf_counter() - t_start:.2f} seconds")

    print("Simulation is over.")

    # Generate the results
    print("Generating the results post-simulation...")
    results = mod5_results(
        dimensions, types, parameters, activities, technologies, agents, policies, results, output_path, plot_price_duration
    )
    print("Results generated successfully.")

    # Note: The commented-out part saves output to .mat file (if interaction with matlab is desired)
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

        # Same output, as relational duckDB databases instead of a pickle:
        # one mirroring the full state above, one mirroring the Excel reports.
        state_db_path = str(Path(output_path) / 'simulation_state.duckdb')
        save_state_duckdb(dimensions, parameters, types, activities, technologies, profiles, results, agents, state_db_path)

        excel_db_path = str(Path(output_path) / 'simulation_excel.duckdb')
        save_excel_duckdb(types, activities, technologies, agents, results, excel_db_path)


    # Print the last time stamp
    print(f"Total time elapsed: {time.perf_counter() - t_start:.2f} seconds")
