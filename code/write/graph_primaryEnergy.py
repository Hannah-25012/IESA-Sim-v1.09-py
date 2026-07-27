# Function to graph the evolution of primary energy
import math
import os
import plotly.graph_objects as go
from plot_colors import rgb

def graph_primaryEnergy(dimensions, types, activities, results, font_name, font_size, color_code, graphs_dir):

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

    # Creating the graph. barmode='relative' stacks each period's positive
    # values upward from 0 and negative values downward from 0 independently
    # (Plotly does this per x-value automatically), which is exactly the
    # positive/negative stacked-bar split the matplotlib version built by
    # hand with separate y1/y2 arrays.
    fig = go.Figure()
    for i in range(nEl):
        idx = order[i]
        fig.add_trace(go.Bar(
            x=list(periods), y=primary_energy[idx, :],
            name=energy_labels[idx],
            marker_color=rgb(color_code, i),
        ))

    fig.update_layout(
        barmode='relative',
        template='plotly_white',
        font=dict(family=font_name, size=font_size),
        yaxis_title='primary energy source [PJ]',
        xaxis=dict(tickmode='array', tickvals=list(periods), range=[2015, 2055]),
        legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5),
        margin=dict(t=30),
    )
    fig.update_yaxes(gridcolor='#e0e0e0', zerolinecolor='#bbb')

    fig.write_html(os.path.join(graphs_dir, 'primary_energy.html'), include_plotlyjs='cdn')
