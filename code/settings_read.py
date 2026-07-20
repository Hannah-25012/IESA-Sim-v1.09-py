# File to read setting from JSON settings file
import json
import os

def settings_read(version_number):

    # Define file name
    json_settings_file = f'settings/IESA_settings_v{version_number}.json'
    
    # Read file
    with open(json_settings_file, 'r') as file:
        json_settings_str = file.read()
    
    json_settings = json.loads(json_settings_str)

    # Decompose the JSON struct
    file_name = json_settings['file_name']
    scenario_name = json_settings['scenario_name']
    read_input = json_settings['read_input']
    # Which duckDB file to read from/write to - optional, defaults to
    # SIMmodel.duckdb in the working directory when absent from the settings
    # file (see main.py). Lets a run be pointed at a specific saved database
    # snapshot instead of always the one in the working directory.
    db_path = json_settings.get('db_path')
    save_output = json_settings['save_output']
    plot_price_duration = json_settings['plot_price_duration']
    nIp = json_settings['nIp']  # Number of power iterations
    nIb = json_settings['nIb']  # Number of balancing iterations
    nId = json_settings['nId']  # Number of dispatch iterations
    year_end = json_settings['year_end']  # Run until period

    # Manage paths
    outputName_root = f'{scenario_name}_Output_v{version_number}'
    input_file = os.path.join('input', file_name)
    output_path = os.path.join('../output', outputName_root)

    # Save settings
    settings = {
        'input': input_file,
        'scenario_name' : scenario_name,
        'output': output_path,
        'read_input': read_input,
        'db_path': db_path,
        'save_output': save_output,
        'plot_price_duration': plot_price_duration,
        'iterations': {
            'power': nIp,
            'balancing': nIb,
            'dispatch': nId
        },
        'year_end': year_end
    }

    return settings