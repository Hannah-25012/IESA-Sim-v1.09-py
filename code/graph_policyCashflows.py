import numpy as np
import matplotlib.pyplot as plt

def graph_policyCashflows(types, activities, results, font_name, font_size, color_code):
    # Extract parameters
    policy_cashflows_categories = types['policy_cashflows_categories']
    periods = activities['periods']
    policy_cashflows = results['policy_cashflows'] / 1000.0

    # Preparing the graph
    policy_cashflows_pos = np.copy(policy_cashflows)
    policy_cashflows_neg = np.copy(policy_cashflows)
    policy_cashflows_pos[policy_cashflows < 0] = 0
    policy_cashflows_neg[policy_cashflows > 0] = 0
    y1 = policy_cashflows_pos
    y2 = policy_cashflows_neg
    lbl = list(policy_cashflows_categories)
    lbl.append('Total')

    # Creating the graph
    _ , ax = plt.subplots()
    # Plot stacked positive bars
    stack_bottom = np.zeros_like(periods, dtype=float)
    bars_a = []
    # Assuming y1 is a 2D array with shape (num_categories, num_periods)
    for i in range(y1.shape[0]):
        bar_container = ax.bar(periods, y1[i, :], bottom=stack_bottom, edgecolor='none')
        stack_bottom += y1[i, :]
        bars_a.append(bar_container)
    # Plot total line over positive bars
    total_line = np.sum(y1, axis=0) + np.sum(y2, axis=0)
    ax.plot(periods, total_line, 'k-', linewidth=2)
    # Plot stacked negative bars
    stack_bottom_neg = np.zeros_like(periods, dtype=float)
    bars_b = []
    for i in range(y2.shape[0]):
        bar_container = ax.bar(periods, y2[i, :], bottom=stack_bottom_neg, edgecolor='none')
        stack_bottom_neg += y2[i, :]
        bars_b.append(bar_container)
    # Plot total line again over negative bars
    ax.plot(periods, total_line, 'k-', linewidth=2)
    ax.set_ylabel('policy cashflows [B€]', fontname=font_name, fontsize=font_size)
    ax.set_ylim([-100, 100])
    
    # Formatting section
    ax.yaxis.grid(True)
    ax.set_xticks(periods)
    ax.set_xticklabels(periods, fontname=font_name, fontsize=font_size)
    plt.xticks(rotation=0)
    ax.set_xlim([2015, 2055])
    # Update x and y tick label font properties
    for label in ax.get_xticklabels():
        label.set_fontname(font_name)
        label.set_fontsize(font_size)
    for label in ax.get_yticklabels():
        label.set_fontname(font_name)
        label.set_fontsize(font_size)
    if True:
        ax.legend(lbl, prop={'family': font_name, 'size': 12}, ncol=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Coloring section
    # Note: Matlab indices start at 1; Python indices are 0-based.
    if len(bars_a) >= 4:
        for patch in bars_a[0]:
            patch.set_facecolor(color_code[11, :])
        for patch in bars_a[1]:
            patch.set_facecolor(color_code[15, :])
        for patch in bars_a[2]:
            patch.set_facecolor(color_code[5, :])
        for patch in bars_a[3]:
            patch.set_facecolor(color_code[6, :])
    if len(bars_b) >= 4:
        for patch in bars_b[0]:
            patch.set_facecolor(color_code[11, :])
        for patch in bars_b[1]:
            patch.set_facecolor(color_code[15, :])
        for patch in bars_b[2]:
            patch.set_facecolor(color_code[5, :])
        for patch in bars_b[3]:
            patch.set_facecolor(color_code[6, :])
    
    # Show the plot
    plt.show(block=False)