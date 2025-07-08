import numpy as np
import matplotlib.pyplot as plt

# Function to graph the evolution of primary energy

def graph_systemEmissions(activities, technologies, results, font_name, font_size):
    
    # Extract parameters
    periods = activities['periods']
    tech_balancers = technologies['balancers']['ids']
    tech_stock_max = technologies['balancers']['stocks']['max']
    emissions = results['emissions']

    # Historical NL emissions
    years = np.arange(2015, 2024)  # creates an array from 2015 to 2023
    realized_emissions = [194.4, 195.1, 192.4, 187.2, 181.4, 164.8, 167.7, 154.1, 145.4]

    # Calculating the emission target
    tech_balancers_arr = np.array(tech_balancers)
    coord_tech = (tech_balancers_arr == 'Emi02_01') | (tech_balancers_arr == 'Emi03_01')
    emission_target = np.sum(tech_stock_max[coord_tech, :], axis=0)

    # Creating the graph
    plt.figure()
    plt.plot(periods, emissions, 'k', linewidth=2)
    plt.plot(periods[2:], emission_target[2:], '--r', linewidth=2)
    plt.plot(years, realized_emissions, ':b', linewidth=2)
    plt.ylabel('system emisisons [Mton CO_2/y]', fontname=font_name, fontsize=font_size)
    plt.legend(['modeled emissions', 'target emissions', 'historical emissions'], prop={'family': font_name, 'size': 12})
    
    plt.show(block=False)
