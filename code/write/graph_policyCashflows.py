# Function to graph policy cashflows
import os
import numpy as np
import plotly.graph_objects as go
from plot_colors import rgb

def graph_policyCashflows(types, activities, results, font_name, font_size, color_code, graphs_dir):

    # Extract parameters
    policy_cashflows_categories = types.policy_cashflows_categories
    periods = activities.periods
    policy_cashflows = results.policy_cashflows / 1000.0

    lbl = list(policy_cashflows_categories)
    num_categories = policy_cashflows.shape[0]

    # Only the first 4 categories (EUA, Taxes, Feed-In subsidies, Investment
    # subsidies) get an explicit color, matching the matplotlib version's own
    # "if len(bars) >= 4" coloring section - any category beyond that keeps
    # Plotly's default color cycle instead of erroring.
    category_colors = [11, 15, 5, 6]

    total_line = np.sum(policy_cashflows, axis=0)

    # Creating the graph. barmode='relative' stacks each period's positive
    # categories upward from 0 and negative ones downward from 0
    # independently, replacing the matplotlib version's hand-split y1/y2.
    fig = go.Figure()
    for i in range(num_categories):
        color = rgb(color_code, category_colors[i]) if i < len(category_colors) else None
        fig.add_trace(go.Bar(
            x=list(periods), y=policy_cashflows[i, :], name=lbl[i],
            marker_color=color,
        ))

    fig.add_trace(go.Scatter(
        x=list(periods), y=list(total_line),
        mode='lines', name='Total', line=dict(color='black', width=2),
    ))

    fig.update_layout(
        barmode='relative',
        template='plotly_white',
        font=dict(family=font_name, size=font_size),
        yaxis_title='policy cashflows [B€]',
        yaxis=dict(range=[-100, 100], tick0=-100, dtick=20),
        xaxis=dict(tickmode='array', tickvals=list(periods), range=[2015, 2055]),
        legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5),
        margin=dict(t=30),
    )
    fig.update_yaxes(gridcolor='#e0e0e0', zerolinecolor='#bbb')

    fig.write_html(os.path.join(graphs_dir, 'policy_cashflows.html'), include_plotlyjs='cdn')
