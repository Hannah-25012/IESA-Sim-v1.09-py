# Function to graph sectoral emissions
# CHECK: Have to go over all of the graphing functions
import numpy as np
import matplotlib.pyplot as plt

def graph_sectoralEmissions(activities, types, results, font_name, font_size, color_code):

    # Extract parameters
    periods = activities.periods
    sectors = types.sectors
    emissions_sector_pos = results.emissions_sector_pos
    emissions_sector_neg = results.emissions_sector_neg
    emissions_stored = results.emissions_stored

    # Order the graph
    nS = len(sectors)
    ordered_labels = ['Residential', 'Services', 'Agriculture', 'Transport', 'Industry',
                      'Power NL', 'Refineries', 'Final Gas', 'Hydrogen', 'Ammonia', 'CCUS', 'nER GHG', 'Others']

    # Name -> position lookup (1-indexed, matching MATLAB), resolved once
    # instead of re-scanning ordered_labels for every sector.
    ordered_idx_by_name = {name: i + 1 for i, name in enumerate(ordered_labels)}
    order = np.zeros(nS, dtype=int)
    for iS in range(nS):
        order[iS] = ordered_idx_by_name.get(sectors[iS], len(ordered_labels))

    # Preparing the graph content
    num_periods = periods.shape[0] if periods.ndim > 0 else len(periods)
    y1_list = []
    y2_list = []
    for iL in range(1, len(ordered_labels) + 1):
        if np.any(order == iL):
            pos_sum = np.sum(emissions_sector_pos[order == iL, :], axis=0)
            neg_sum = np.sum(emissions_sector_neg[order == iL, :], axis=0)
        else:
            pos_sum = np.zeros(num_periods)
            neg_sum = np.zeros(num_periods)
        y1_list.append(pos_sum)
        y2_list.append(neg_sum)
    y1 = np.vstack(y1_list)  # rows correspond to each ordered label (length = len(ordered_labels))
    y2 = np.vstack(y2_list)

    # Append emissions_stored to y1
    y1 = np.vstack([y1, emissions_stored])
    lbl = ordered_labels.copy()
    lbl.append('Stored CO_2')
    lbl.append('Total Emissions')
    lbl.append('Total CO_2')

    # Creating the graph (plot y1 and y2 as stacked bars)
    # y1
    _ , ax = plt.subplots()
    bars_a = []
    bottom = np.zeros_like(periods, dtype=float)
    for i in range(y1.shape[0]):
        bar_container = ax.bar(periods, y1[i, :], bottom=bottom, edgecolor='none')
        bars_a.append(bar_container)
        bottom = bottom + y1[i, :]

    ax.plot(periods, np.sum(y1, axis=0) + np.sum(y2, axis=0) - emissions_stored, 'k', linewidth=2) # Plot the two lines
    ax.plot(periods, np.sum(y1, axis=0) + np.sum(y2, axis=0), '--k', linewidth=2)

   # y2
    bars_b = []
    bottom = np.zeros_like(periods, dtype=float)
    for i in range(y2.shape[0]):
        bar_container = ax.bar(periods, y2[i, :], bottom=bottom, edgecolor='none')
        bars_b.append(bar_container)
        bottom = bottom + y2[i, :]

    ax.set_ylabel('sectoral emissions [Mton/y]', fontname=font_name, fontsize=font_size)
    ax.set_ylim([-100, 200])

    # Formatting section
    ax.yaxis.grid(True)
    ax.set_xticks(periods)
    ax.set_xticklabels(periods, fontname=font_name, fontsize=font_size)
    plt.xticks(rotation=0)
    ax.set_xlim([2015, 2055])
    for label in ax.get_xticklabels():
        label.set_fontname(font_name)
        label.set_fontsize(font_size)
    for label in ax.get_yticklabels():
        label.set_fontname(font_name)
        label.set_fontsize(font_size)
    if True:
        ax.legend(lbl, prop={'family': font_name, 'size': 12}, ncol=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Coloring section
    # For bars_a (y1)
    a_colors = [color_code[11], color_code[9], color_code[5], color_code[2],
                color_code[0], color_code[10], color_code[12], color_code[1],
                color_code[7], color_code[14], color_code[15], color_code[4],
                color_code[3], color_code[8]]
    for i, bar_container in enumerate(bars_a):
        for rect in bar_container:
            rect.set_facecolor(a_colors[i])
            if i == 13:  # the 14th element, set alpha to 0.5
                rect.set_alpha(0.5)

    # For bars_b (y2)
    b_colors = [color_code[11], color_code[9], color_code[5], color_code[2],
                color_code[0], color_code[10], color_code[12], color_code[1],
                color_code[7], color_code[14], color_code[15], color_code[4],
                color_code[3]]
    for i, bar_container in enumerate(bars_b):
        for rect in bar_container:
            rect.set_facecolor(b_colors[i])

    plt.show(block=False)
