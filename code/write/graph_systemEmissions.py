# Function to graph the evolution of system emissions
import os
import numpy as np
import plotly.graph_objects as go

def graph_systemEmissions(activities, technologies, results, font_name, font_size, graphs_dir):

    # Extract parameters
    periods = activities.periods
    tech_entities = technologies.balancers.entities
    tech_stock_max = technologies.balancers.stocks.max
    emissions = results.emissions

    # Historical NL emissions
    years = np.arange(2015, 2024)  # creates an array from 2015 to 2023
    realized_emissions = [194.4, 195.1, 192.4, 187.2, 181.4, 164.8, 167.7, 154.1, 145.4]

    # Calculating the emission target
    tech_by_id = {tech.id: tech for tech in tech_entities}
    target_techs = [tech_by_id[tid] for tid in ('Emi02_01', 'Emi03_01') if tid in tech_by_id]
    coord_tech = [tech.idx for tech in target_techs]
    emission_target = np.sum(tech_stock_max[coord_tech, :], axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(periods), y=list(emissions), mode='lines',
                              name='modeled emissions', line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=list(periods[2:]), y=list(emission_target[2:]), mode='lines',
                              name='target emissions', line=dict(color='red', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=list(years), y=realized_emissions, mode='lines',
                              name='historical emissions', line=dict(color='blue', width=2, dash='dot')))

    fig.update_layout(
        template='plotly_white',
        font=dict(family=font_name, size=font_size),
        yaxis_title='system emisisons [Mton CO_2/y]',
        margin=dict(t=30),
    )
    fig.update_yaxes(gridcolor='#e0e0e0')

    fig.write_html(os.path.join(graphs_dir, 'system_emissions.html'), include_plotlyjs='cdn')
