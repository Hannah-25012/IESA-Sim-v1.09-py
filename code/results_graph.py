from graph_primaryEnergy import graph_primaryEnergy
from graph_systemEmissions import graph_systemEmissions
from graph_systemCosts import graph_systemCosts
from graph_sectoralEmissions import graph_sectoralEmissions
from graph_policyCashflows import graph_policyCashflows
from graph_priceDuration import graph_priceDuration


import numpy as np
import matplotlib as mpl


def results_graph(dimensions, types, activities, technologies, results, plot_price_duration):
    # Unify the format and colors
    # Plot format
    
    font_name = 'DejaVu Sans'
    font_size = 12
    
    mpl.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 12,
    })
    # font_name = 'Helvetica'
    # font_size = 12

    # Color code
    # In Matlab, zeros(13,3) is used but then rows 14-18 are added.
    # Here we initialize an array with 18 rows to accommodate all entries.
    color_code = np.zeros((18, 3))
    color_code[0, :] = [0.0, 0.0, 0.0]   # Black
    color_code[1, :] = [0.3, 0.3, 0.3]   # Gray dark
    color_code[2, :] = [0.5, 0.5, 0.5]   # Gray light
    color_code[3, :] = [0.5, 0.0, 1.0]   # Purple
    color_code[4, :] = [0.5, 0.3, 0.2]   # Brown
    color_code[5, :] = [0.3, 0.8, 0.3]   # Green
    color_code[6, :] = [0.0, 0.5, 0.0]   # Dark Green
    color_code[7, :] = [0.7, 0.9, 1.0]   # Light blue
    color_code[8, :] = [0.9, 0.9, 0.0]   # Yellow
    color_code[9, :] = [0.0, 0.2, 0.6]   # Blue
    color_code[10, :] = [1.0, 0.5, 0.0]  # Orange
    color_code[11, :] = [0.9, 0.0, 0.0]  # Red
    color_code[12, :] = [1.0, 0.5, 0.8]  # Pink
    color_code[13, :] = [0.4, 0.5, 0.6]  # Grayblue
    color_code[14, :] = [0.5, 0.8, 0.7]  # Aqua
    color_code[15, :] = [0.6, 0.0, 0.0]  # Dark Red
    color_code[16, :] = [0.8, 0.0, 0.8]  # Fuschia
    color_code[17, :] = [0.6, 0.8, 0.1]  # Lime
    # Contrasting color order
    # contrast_order = [10, 16, 6, 4, 11, 15, 13, 1, 8, 9, 2, 17, 2, 12, 3, 7, 5]

    # Graph the primary energy balance
    print("--Graphing the primary energy balance...")
    graph_primaryEnergy(dimensions, types, activities, results, font_name, font_size, color_code)

    # Graph the total emissions
    print("--Graphing the system GHG emissions...")
    graph_systemEmissions(activities, technologies, results, font_name, font_size)

    # Graph the system costs
    print("--Graphing the system costs...")
    graph_systemCosts(activities, results, font_name, font_size, color_code)

    # Graph the sectoral emissions
    print("--Graphing the sectoral emissions...")
    graph_sectoralEmissions(activities, types, results, font_name, font_size, color_code)

    # Graph the policy cashflows
    print("--Graphing the policy cashflows")
    graph_policyCashflows(types, activities, results, font_name, font_size, color_code)

    # Graph the power price duration curves if requested
    if plot_price_duration:
        print("--Graphing the price duration curves...")
        graph_priceDuration(activities, font_name, font_size, color_code)
