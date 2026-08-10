# Function to graph sectoral emissions
import os
import numpy as np
import plotly.graph_objects as go
from plot_colors import rgb

def graph_sectoralEmissions(activities, types, results, font_name, font_size, color_code, graphs_dir):

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

    # Coloring section - same per-sector color on both the positive and
    # negative stack (b_colors is a_colors minus its last "Stored CO2" entry,
    # which has no negative-side counterpart at all).
    a_colors = [11, 9, 5, 2, 0, 10, 12, 1, 7, 14, 15, 4, 3, 8]
    b_colors = a_colors[:13]

    # Creating the graph. Each sector's positive (gross-emitting) and negative
    # (net-reducing, e.g. CCUS-abated) stacks are genuinely different
    # quantities sharing one color, so each gets its own labeled legend entry
    # ("<sector> +" / "<sector> -") instead of the negative side hiding behind
    # the positive one under a shared name. Stored CO_2 has no negative-side
    # counterpart at all, so it keeps its plain name.
    fig = go.Figure()
    for i in range(y1.shape[0]):
        is_stored = i == len(ordered_labels)
        name = 'Stored CO_2' if is_stored else f'{ordered_labels[i]} +'
        fig.add_trace(go.Bar(
            x=list(periods), y=y1[i, :], name=name,
            marker_color=rgb(color_code, a_colors[i]),
            marker_opacity=0.5 if is_stored else 1.0,
        ))
    for i in range(y2.shape[0]):
        fig.add_trace(go.Bar(
            x=list(periods), y=y2[i, :], name=f'{ordered_labels[i]} -',
            marker_color=rgb(color_code, b_colors[i]),
        ))

    fig.add_trace(go.Scatter(
        x=list(periods), y=list(np.sum(y1, axis=0) + np.sum(y2, axis=0) - emissions_stored),
        mode='lines', name='Total Emissions', line=dict(color='black', width=2),
    ))
    fig.add_trace(go.Scatter(
        x=list(periods), y=list(np.sum(y1, axis=0) + np.sum(y2, axis=0)),
        mode='lines', name='Total CO_2', line=dict(color='black', width=2, dash='dash'),
    ))

    fig.update_layout(
        barmode='relative',
        template='plotly_white',
        font=dict(family=font_name, size=font_size),
        yaxis_title='sectoral emissions [Mton/y]',
        yaxis_range=[-100, 200],
        xaxis=dict(tickmode='array', tickvals=list(periods), range=[2015, 2055]),
        legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5),
        margin=dict(t=30),
    )
    fig.update_yaxes(gridcolor='#e0e0e0', zerolinecolor='#bbb')

    fig.write_html(os.path.join(graphs_dir, 'sectoral_emissions.html'), include_plotlyjs='cdn')
