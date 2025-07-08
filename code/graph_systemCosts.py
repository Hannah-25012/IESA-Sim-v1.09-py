import numpy as np
import matplotlib.pyplot as plt

def graph_systemCosts(activities, results, font_name, font_size, color_code):
    # Export Parameters
    periods = activities['periods']
    cost_categories = results['costs']['categories']
    system_costs = results['costs']['system']

    # Order the graph
    ordered_label = cost_categories

    # Clean the nans (if there are any for test runs)
    # system_costs[np.isnan(system_costs)] = 0  # Clean NaNs if any

    # Preparing the graph
    system_costspos = system_costs.copy()
    system_costsneg = system_costs.copy()
    system_costspos[system_costs < 0] = 0
    system_costsneg[system_costs > 0] = 0
    y1 = system_costspos.T
    y2 = system_costsneg.T
    lbl = ordered_label

    # Creating the graph
    plt.figure()
    # "hold on" is implicit in matplotlib

    # Plot positive bars with stacking
    a = []
    num_categories = y1.shape[1]
    for i in range(num_categories):
        if i == 0:
            bottom = np.zeros_like(periods, dtype=float)
        else:
            bottom = np.sum(y1[:, :i], axis=1)
        container = plt.bar(periods, y1[:, i] / 1000, bottom=bottom / 1000, edgecolor='none')
        a.append(container)

    # Plot the line (sum over cost categories)
    plt.plot(periods, np.sum(system_costs, axis=0) / 1000, 'k', linewidth=2)

    # Plot negative bars with stacking
    b = []
    for i in range(num_categories):
        if i == 0:
            bottom = np.zeros_like(periods, dtype=float)
        else:
            bottom = np.sum(y2[:, :i], axis=1)
        container = plt.bar(periods, y2[:, i] / 1000, bottom=bottom / 1000, edgecolor='none')
        b.append(container)

    plt.ylabel('system costs [B€/y]', fontname=font_name, fontsize=font_size)
    plt.ylim([-150, 250])
    # "hold off" is implicit when plotting is complete

    # Formatting section
    ax = plt.gca()
    ax.yaxis.grid(True)
    plt.xticks(periods, periods, rotation=0, fontname=font_name, fontsize=font_size)
    plt.xlim(2015, 2055)
    plt.yticks(fontname=font_name, fontsize=font_size)
    # Always display legend (if true)
    plt.legend(lbl, prop={'family': font_name, 'size': 12}, ncol=3)
    # Remove top and right box borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Coloring section
    # Mapping: MATLAB indices [2, 8, 6, 9, 11, 4] become Python indices [1, 7, 5, 8, 10, 3]
    mapping = [1, 7, 5, 8, 10, 3]
    for i, container in enumerate(a):
        for patch in container.patches:
            patch.set_facecolor(color_code[mapping[i]])
    for i, container in enumerate(b):
        for patch in container.patches:
            patch.set_facecolor(color_code[mapping[i]])

    # (Optionally, show the plot or return the figure/axes)
    plt.show(block=False)
