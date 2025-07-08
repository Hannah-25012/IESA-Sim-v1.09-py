# Determine the investment potential per technology
# Attention: Verify that the variables tech_stock_deploy and retrofit_potential are 1-dimensional arrays (and not 2-dimensional)
# Attention: the logic of the code seems to prioritize deployment first, then retrofitting. Also, I don't exactly understand what happens with room_to_invest if retrofit option is chosen. Check for-loop again.

import numpy as np

def invest_investment_potential(dimensions, technologies, tech_stock_exist, retrofit_potential, ip):
    
    # Extract parameters
    nTb = dimensions['nTb']
    tech_stock_deploy = technologies['balancers']['stocks']['deploy']
    tech_stock_max = technologies['balancers']['stocks']['max'][:, ip]

    # Obtain investment potentials
    investment_potential = np.zeros(nTb)  # Preallocate
    for iTb in range(nTb):
        # Obtain the raw room to invest
        room_to_invest = tech_stock_max[iTb] - tech_stock_exist[iTb]

        # Check if there is a deploy limitation or a retrofit option
        if tech_stock_deploy[iTb] > 0:
            investment_potential[iTb] = min(tech_stock_deploy[iTb], room_to_invest)
        elif retrofit_potential[iTb] > 0:
            investment_potential[iTb] = retrofit_potential[iTb]
        else:
            investment_potential[iTb] = room_to_invest

    return investment_potential

