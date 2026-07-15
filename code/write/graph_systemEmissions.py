# Function to graph the evolution of system emissions
# CHECK: Have to go over all graphing functions
import numpy as np
import matplotlib.pyplot as plt

def graph_systemEmissions(activities, technologies, results, font_name, font_size):

    # Extract parameters
    periods = activities.periods
    tech_entities = technologies.balancers.entities
    tech_stock_max = technologies.balancers.stocks.max
    emissions = results.emissions

    # Historical NL emissions
    years = np.arange(2015, 2024)  # creates an array from 2015 to 2023
    realized_emissions = [194.4, 195.1, 192.4, 187.2, 181.4, 164.8, 167.7, 154.1, 145.4]

    # Calculating the emission target
    tech_by_id = {tech.id: tech for tech in tech_entities}
    target_techs = [tech_by_id[tid] for tid in ('Emi02_01', 'Emi03_01') if tid in tech_by_id]
    coord_tech = [tech.idx for tech in target_techs]
    emission_target = np.sum(tech_stock_max[coord_tech, :], axis=0)

    # Creating the graph
    plt.figure()
    plt.plot(periods, emissions, 'k', linewidth=2)
    plt.plot(periods[2:], emission_target[2:], '--r', linewidth=2)
    plt.plot(years, realized_emissions, ':b', linewidth=2)
    plt.ylabel('system emisisons [Mton CO_2/y]', fontname=font_name, fontsize=font_size)
    plt.legend(['modeled emissions', 'target emissions', 'historical emissions'], prop={'family': font_name, 'size': 12})

    plt.show(block=False)
