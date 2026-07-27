# Function to graph the evolution of system costs
import os
import numpy as np
import plotly.graph_objects as go
from plot_colors import rgb

def graph_systemCosts(activities, results, font_name, font_size, color_code, graphs_dir):

    # Export Parameters
    periods = activities.periods
    cost_categories = results.costs.categories
    system_costs = results.costs.system

    lbl = cost_categories
    num_categories = system_costs.shape[0]

    # MATLAB indices [2, 8, 6, 9, 11, 4] become Python indices [1, 7, 5, 8, 10, 3]
    mapping = [1, 7, 5, 8, 10, 3]

    # Creating the graph. barmode='relative' stacks each period's positive
    # categories upward from 0 and negative ones downward from 0
    # independently, replacing the matplotlib version's hand-split y1/y2.
    fig = go.Figure()
    for i in range(num_categories):
        fig.add_trace(go.Bar(
            x=list(periods), y=system_costs[i, :] / 1000,
            name=lbl[i], marker_color=rgb(color_code, mapping[i]),
        ))

    # The matplotlib version's legend accidentally mislabeled this line
    # 'capital' (a quirk of matplotlib's positional label-to-artist zip
    # picking the line before some of the bars) even though it's the sum of
    # every cost category, capital included - name it for what it actually is.
    fig.add_trace(go.Scatter(
        x=list(periods), y=list(np.sum(system_costs, axis=0) / 1000),
        mode='lines', name='Total', line=dict(color='black', width=2),
    ))

    fig.update_layout(
        barmode='relative',
        template='plotly_white',
        font=dict(family=font_name, size=font_size),
        yaxis_title='system costs [B€/y]',
        xaxis=dict(tickmode='array', tickvals=list(periods), range=[2015, 2055]),
        legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5),
        margin=dict(t=30),
    )
    fig.update_yaxes(gridcolor='#e0e0e0', zerolinecolor='#bbb')

    fig.write_html(os.path.join(graphs_dir, 'system_costs.html'), include_plotlyjs='cdn')
