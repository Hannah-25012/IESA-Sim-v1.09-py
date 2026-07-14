# File to plot the price duration curves
# CHECK: Have to go over all graphing routines again
import numpy as np
import matplotlib.pyplot as plt

def graph_priceDuration(activities, font_name, font_size, color_code):

    # Extract parameters
    periods = activities['periods']
    activities_elec = activities['electricity']['names']
    activities_elec_coord = activities['electricity']['coords']
    prices_hourly = activities['prices']['hourly'] [:, activities_elec_coord, :]

    # For each activity prepare price duration of every year
    nP = len(periods)
    nAk = len(activities_elec)
    nH = prices_hourly.shape[0]
    price_duration = np.zeros((nH, nP, nAk))
    for iAk in range(nAk):
        for iP in range(nP):
            price_duration[:, iP, iAk] = np.sort(prices_hourly[:, iAk, iP]) # Sort each column in ascending order
    
    # For each activity prepare a new plot with the evolution of the price duration curve
    x_ticks = np.linspace(0, nH, num=11)  # creates ticks: 0, n_h/10, ..., n_h
    x_ticks_lbls = list(range(0, 101, 10))
    lbl = [str(p) for p in periods]
    max_price = 300
    color_order = [16, 12, 11, 9, 18, 6, 7]  # Note: color_order values are 1-indexed in MATLAB; here we subtract 1 when indexing color_code.

    for iAk in range(nAk):

        # Graph creation section
        plt.figure()
        lines = plt.plot(np.arange(1, nH + 1), price_duration[:, :, iAk] * 3.6, linewidth=2) # Plot each column of price_duration for the given activity and multiply by 3.6
        y_lbl = f"{activities_elec[iAk]} - price duration in €/MWh"
        plt.ylabel(y_lbl, fontname=font_name, fontsize=font_size)
        plt.xlabel('time duration in %', fontname=font_name, fontsize=font_size)
        
        # Formatting section
        ax = plt.gca()
        ax.yaxis.grid(True)
        plt.xticks(x_ticks, x_ticks_lbls, rotation=0, fontname=font_name, fontsize=font_size)
        plt.xlim(0, nH)
        plt.ylim(0, max_price)
        xticks_labels = [item.get_text() for item in ax.get_xticklabels()] # Reapply tick labels with the desired font settings
        yticks_labels = [item.get_text() for item in ax.get_yticklabels()]
        ax.set_xticklabels(xticks_labels, fontname=font_name, fontsize=font_size)
        ax.set_yticklabels(yticks_labels, fontname=font_name, fontsize=font_size)
        plt.legend(lbl, prop={'family': font_name, 'size': 12}, loc='best')
        ax.spines['top'].set_visible(False) # Remove top and right spines to mimic Matlab 'box off'
        ax.spines['right'].set_visible(False)
        
        # Coloring section
        for iP in range(nP):
            lines[iP].set_color(color_code[color_order[iP] - 1]) # Subtract 1 from color_order value to convert MATLAB 1-indexing to Python's 0-indexing
        
        plt.show(block=False)