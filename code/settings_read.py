import json
import os

def settings_read(version_number):
    # Read settings from the JSON settings file
    json_settings_file = f'IESA_settings_v{version_number}.json'
    
    with open(json_settings_file, 'r') as file:
        json_settings_str = file.read()
    
    json_settings = json.loads(json_settings_str)

    # Decompose the JSON struct
    file_name = json_settings['file_name']
    scenario_name = json_settings['scenario_name']
    read_input = json_settings['read_input']
    save_output = json_settings['save_output']
    plot_price_duration = json_settings['plot_price_duration']
    nIp = json_settings['nIp']  # Number of power iterations
    nIb = json_settings['nIb']  # Number of balancing iterations
    nId = json_settings['nId']  # Number of dispatch iterations
    year_end = json_settings['year_end']  # Run until period

    # Manage paths
    outputName_root = f'{scenario_name}_Output_v{version_number}'
    input_file = os.path.join('input', file_name)
    output_path = os.path.join('output', outputName_root)

    # Save settings
    settings = {
        'input': input_file,
        'scenario_name' : scenario_name,
        'output': output_path,
        'read_input': read_input,
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