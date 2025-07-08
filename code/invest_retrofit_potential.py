# Function to determine the retrofitting potential per technology

import numpy as np

def invest_retrofit_potential(dimensions, technologies, tech_stock_exist):
    
    # Extract parameters
    nTb = dimensions['nTb']
    tech_balancers = technologies['balancers']['ids']
    retrofits_to = technologies['retrofittings']['to']
    retrofits_from = technologies['retrofittings']['from']
    retrofits_costs = technologies['retrofittings']['costs']

    # Preallocate
    retrofit_sources = [[None] * 15 for _ in range(nTb)]
    retrofit_options = np.zeros(nTb, dtype=int)
    retrofit_potential = np.zeros(nTb, dtype=float)
    # Attention: According to Vinzenz adding a very high number isn't the most elegant way, maybe it can be changed?
    retrofit_cost = np.full(nTb, 1e9, dtype=float)  # High initial value for minimum tracking (Manuel says: % High enough number to then store the min)

    for iTb in range(nTb):
        # Find the retrofitting options of the technology
        coord_to = [retrofit == tech_balancers[iTb] for retrofit in retrofits_to]
        n_opts = sum(coord_to)
        n_froms = 0

        # If there are any options, proceed
        if n_opts > 0:
            # Identify the available stock per retrofit alternative
            options_name = [retrofits_from[idx] for idx, val in enumerate(coord_to) if val]
            options_costs = [retrofits_costs[idx] for idx, val in enumerate(coord_to) if val]

            for i_opts in range(n_opts):
                # Identify the original technology
                coord_from = [tech == options_name[i_opts] for tech in tech_balancers]

                # Check if there is available stock
                av_stock = tech_stock_exist[coord_from].sum()
                if av_stock > 0:
                    # Confirm the option
                    n_froms += 1

                    # Increase the potential
                    retrofit_potential[iTb] += av_stock

                    # Store the minimal cost
                    retrofit_cost[iTb] = min(retrofit_cost[iTb], options_costs[i_opts])

                    # Store the name of the retrofit source
                    retrofit_sources[iTb][n_froms - 1] = options_name[i_opts]

            # Save the number of retrofit options
            retrofit_options[iTb] = n_froms

    # Shrink the retrofit sources list
    n_opts_max = max(retrofit_options)
    retrofit_sources = [row[:n_opts_max] for row in retrofit_sources]

    return retrofit_sources, retrofit_options, retrofit_potential, retrofit_cost
