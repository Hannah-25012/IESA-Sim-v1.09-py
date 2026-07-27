# File to plot the price duration curves
import os
import re
import numpy as np
import plotly.graph_objects as go
from plot_colors import rgb

def graph_priceDuration(activities, font_name, font_size, color_code, graphs_dir):

    # Extract parameters
    periods = activities.periods
    elec_activities = [a for a in activities.entities if a.is_electricity]
    activities_elec_coord = activities.electricity.coords
    prices_hourly = activities.prices.hourly[:, activities_elec_coord, :]

    # For each activity prepare price duration of every year
    nP = len(periods)
    nAk = len(elec_activities)
    nH = prices_hourly.shape[0]
    price_duration = np.zeros((nH, nP, nAk))
    for iAk in range(nAk):
        for iP in range(nP):
            price_duration[:, iP, iAk] = np.sort(prices_hourly[:, iAk, iP]) # Sort each column in ascending order

    # For each activity prepare a new plot with the evolution of the price duration curve
    x = np.arange(1, nH + 1)
    x_ticks = np.linspace(0, nH, num=11)  # creates ticks: 0, n_h/10, ..., n_h
    x_ticks_lbls = list(range(0, 101, 10))
    lbl = [str(p) for p in periods]
    max_price = 300
    color_order = [16, 12, 11, 9, 18, 6, 7]  # Note: values are 1-indexed in MATLAB; here we subtract 1 when indexing color_code.

    for act in elec_activities:
        iAk = act.elec_idx

        fig = go.Figure()
        for iP in range(nP):
            # color_order only has 7 entries (pre-existing from the matplotlib
            # version, which would IndexError identically on an 8+ period run -
            # this graph is opt-in and off by default, so it was never
            # exercised for real) - wrap around instead of assuming nP <= 7.
            fig.add_trace(go.Scatter(
                x=list(x), y=list(price_duration[:, iP, iAk] * 3.6),
                mode='lines', name=lbl[iP],
                line=dict(color=rgb(color_code, color_order[iP % len(color_order)] - 1), width=2),
            ))

        fig.update_layout(
            template='plotly_white',
            font=dict(family=font_name, size=font_size),
            yaxis_title=f"{act.name} - price duration in €/MWh",
            xaxis_title='time duration in %',
            xaxis=dict(tickmode='array', tickvals=list(x_ticks), ticktext=[str(v) for v in x_ticks_lbls], range=[0, nH]),
            yaxis_range=[0, max_price],
            margin=dict(t=30),
        )
        fig.update_yaxes(gridcolor='#e0e0e0')

        safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', act.name).strip('_')
        fig.write_html(os.path.join(graphs_dir, f'price_duration_{safe_name}.html'), include_plotlyjs='cdn')
