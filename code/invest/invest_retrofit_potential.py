# Function to determine the retrofitting potential per technology
import numpy as np

def invest_retrofit_potential(dimensions, technologies, tech_stock_exist):

    # Extract parameters
    nTb = dimensions['nTb']
    tech_entities = technologies.balancers.entities

    # Determine number of options per retrofittable technologies and potentials
    retrofit_sources = [[None] * 15 for _ in range(nTb)]
    retrofit_options = np.zeros(nTb, dtype=int) # Preallocate
    retrofit_potential = np.zeros(nTb, dtype=float)
    retrofit_cost = np.full(nTb, 1e9, dtype=float)  # High initial value for minimum tracking (Manuel says: % High enough number to then store the min) - # CHECK: According to Vinzenz, adding a very high number isn't the most elegant way, maybe it can be changed?

    for tech in tech_entities:

        # tech.retrofit_sources: (source_tech, cost) pairs this technology can be retrofitted from
        n_froms = 0
        for source_tech, cost in tech.retrofit_sources:

            # Check if there is available stock
            av_stock = tech_stock_exist[source_tech.idx]
            if av_stock > 0:

                # Confirm the option
                n_froms += 1

                # Increase the potential
                retrofit_potential[tech.idx] += av_stock

                # Store the minimal cost
                retrofit_cost[tech.idx] = min(retrofit_cost[tech.idx], cost)

                # Store the retrofit source
                retrofit_sources[tech.idx][n_froms - 1] = source_tech

        # Save the number of retrofit options
        retrofit_options[tech.idx] = n_froms

    # Shrink the retrofit sources list
    n_opts_max = max(retrofit_options)
    retrofit_sources = [row[:n_opts_max] for row in retrofit_sources]

    return retrofit_sources, retrofit_options, retrofit_potential, retrofit_cost
