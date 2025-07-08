import numpy as np
import matplotlib.pyplot as plt
import math

def graph_primaryEnergy(dimensions, types, activities, results, font_name, font_size, color_code):
    # Extract parameters
    nEl = dimensions['nEl']
    energy_labels = types['energy']['labels']
    periods = activities['periods']
    primary_energy = results['primary']

    # Order the graph
    ordered_labels = ['Coal', 'Oil', 'Natural Gas', 'Nuclear', 'Waste', 'Biomass',
                      'Bio-fuels', 'Hydrogen', 'Solar', 'Wind', 'Other RE', 'Electricity',
                      'Synfuels', 'Oil Products', 'Ammonia', 'Heat', 'NA']
    order = []
    # The list so far contains nan as the last value, so we need to replace with 'NA' to match the ordered_labels 
    energy_labels = [x if not (isinstance(x, float) and math.isnan(x)) 
                 else 'NA'
                 for x in energy_labels]
    for i in range(nEl):
        order.append(energy_labels.index(ordered_labels[i]))

    # Preparing the graph
    primary_energy_pos = primary_energy.copy()
    primary_energy_neg = primary_energy.copy()
    primary_energy_pos[primary_energy < 0] = 0
    primary_energy_neg[primary_energy > 0] = 0
    gC = np.abs(np.sum(primary_energy, axis=1)) > 0
    y1 = primary_energy_pos
    y2 = primary_energy_neg
    lbl = [energy_labels[i] for i in order]

    # Creating the graph
    fig, ax = plt.subplots()
    # Reorder data as in Matlab: y1_plot = y1(order,:) and y2_plot = y2(order,:)
    y1_plot = y1[order, :]
    y2_plot = y2[order, :]

    # Plot positive values as a stacked bar chart
    bottom_pos = np.zeros(len(periods))
    bars_positive = []
    for i in range(nEl):
        bar = ax.bar(periods, y1_plot[i, :], bottom=bottom_pos, edgecolor='none',
                     color=color_code[i], label=lbl[i])
        bars_positive.append(bar)
        bottom_pos = bottom_pos + y1_plot[i, :]

    # Plot negative values as a stacked bar chart
    bottom_neg = np.zeros(len(periods))
    bars_negative = []
    for i in range(nEl):
        bar = ax.bar(periods, y2_plot[i, :], bottom=bottom_neg, edgecolor='none',
                     color=color_code[i])
        bars_negative.append(bar)
        bottom_neg = bottom_neg + y2_plot[i, :]

    ax.set_ylabel('primary energy source [PJ]', fontname=font_name, fontsize=font_size)
    ax.set_ylim([-2000, 6000])

    # Formatting section
    ax.yaxis.grid(True)
    ax.set_xticks(periods)
    ax.set_xticklabels(periods, fontname=font_name, fontsize=font_size, rotation=0)
    ax.set_xlim([2015, 2055])
    for tick in ax.get_yticklabels():
        tick.set_fontname(font_name)
        tick.set_fontsize(font_size)
    
    # Legend (horizontal orientation, 5 columns)
    ax.legend(ncol=5, prop={'family': font_name, 'size': 12})
    
    # Remove top and right borders (box off)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.show(block=False)

