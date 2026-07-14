# Determine the investment potential per technology
# THINK: the logic of the code seems to prioritize deployment first, then retrofitting. Also, I don't exactly understand what happens with room_to_invest if retrofit option is chosen
import numpy as np

def invest_investment_potential(dimensions, technologies, tech_stock_exist, retrofit_potential, ip):

    # Extract parameters
    nTb = dimensions['nTb']
    tech_entities = technologies.balancers.entities
    tech_stock_max = technologies.balancers.stocks.max[:, ip]

    # Obtain investment potentials
    investment_potential = np.zeros(nTb)  # Preallocate
    for tech in tech_entities:

        # Obtain the raw room to invest
        room_to_invest = tech_stock_max[tech.idx] - tech_stock_exist[tech.idx]

        # Check if there is a deploy limitation or a retrofit option
        if tech.stock_deploy > 0:
            investment_potential[tech.idx] = min(tech.stock_deploy, room_to_invest)
        elif retrofit_potential[tech.idx] > 0:
            investment_potential[tech.idx] = retrofit_potential[tech.idx]
        else:
            investment_potential[tech.idx] = room_to_invest

    return investment_potential
