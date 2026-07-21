# Function to graph the evolution of primary energy
# CHECK: Have to go over this section to check if positive and neg parts are needed and to make the plot more pretty
import numpy as np
import matplotlib.pyplot as plt
import math

def graph_primaryEnergy(dimensions, types, activities, results, font_name, font_size, color_code):

    # Extract parameters
    nEl = dimensions['nEl']
    energy_labels = types.energy.labels
    periods = activities.periods
    primary_energy = results.primary

    # Order the graph
    ordered_labels = ['Coal', 'Oil', 'Natural Gas', 'Nuclear', 'Waste', 'Biomass',
                      'Bio-fuels', 'Hydrogen', 'Solar', 'Wind', 'Other RE', 'Electricity',
                      'Synfuels', 'Oil Products', 'Ammonia', 'Heat', 'NA']

    # The list so far contains nan as the last value, so we need to replace with 'NA' to match the ordered_labels
    energy_labels = [x if not (isinstance(x, float) and math.isnan(x))
                 else 'NA'
                 for x in energy_labels]

    # Name -> position lookup, resolved once instead of re-scanning energy_labels per ordered label.
    label_idx_by_name = {name: i for i, name in enumerate(energy_labels)}
    order = [label_idx_by_name[ordered_labels[i]] for i in range(nEl)]

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
    # Sized wider/taller than matplotlib's 6.4x4.8 default so the 5-column,
    # up-to-17-entry legend below the axes doesn't get clipped by the figure edge.
    fig, ax = plt.subplots(figsize=(10, 7))
    y1_plot = y1[order, :] # Reorder data as in Matlab: y1_plot = y1(order,:) and y2_plot = y2(order,:)
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

    # Fixed [-2000, 6000] limits (MATLAB-era default) clipped everything above
    # ~6000 PJ - e.g. all of Nuclear/Solar/Wind, stacked higher up - whenever a
    # scenario's total primary energy ran bigger than that. Size to the actual
    # data instead, with a 10% margin so nothing touches the axes edge.
    top = np.sum(y1_plot, axis=0).max()
    bottom = np.sum(y2_plot, axis=0).min()
    margin = 0.1 * (top - bottom)
    ax.set_ylim(bottom - margin, top + margin)

    # Formatting section
    ax.yaxis.grid(True)
    ax.set_xticks(periods)
    ax.set_xticklabels(periods, fontname=font_name, fontsize=font_size, rotation=0)
    ax.set_xlim([2015, 2055])
    for tick in ax.get_yticklabels():
        tick.set_fontname(font_name)
        tick.set_fontsize(font_size)

    ax.legend(ncol=5, prop={'family': font_name, 'size': 12}) # Legend (horizontal orientation, 5 columns)

    ax.spines['top'].set_visible(False) # Remove top and right borders (box off)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    plt.show(block=False)
