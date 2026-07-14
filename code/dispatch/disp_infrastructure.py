# File to determine required infrastructure expansion
import numpy as np

def disp_infrastructure(dimensions, activities, technologies, tech_use_hourly, tech_stock, iP):

    # Extract Parameters
    nTi = dimensions['nTi']
    activity_balances = technologies.balancers.activity_balances
    tech_entities = technologies.balancers.entities
    investments = technologies.balancers.investments[:, iP]
    if iP == 0:
        techstock_exist = technologies.balancers.stocks.initial
    else:
        techstock_exist = technologies.balancers.stocks.evolution[:, iP-1]
    tech_stock_exist_infra = technologies.infra.stocks.initial

    # Identify the network use of activities that have infrastructure
    investments_infra = np.zeros(nTi) # Preallocate
    for activity in activities.entities:
        if not activity.infrastructure:
            continue

        # It is assumed each infra-linked activity has exactly one matching infrastructure entry
        infra_obj = activity.infrastructure[0]

        activity_balances_filtered = -activity_balances[:, activity.idx] # Select and invert the corresponding column
        activity_balances_filtered[activity_balances_filtered > 0] = 0 # Set any positive values to zero
        network_profile = -np.dot(tech_use_hourly, activity_balances_filtered) # Compute the network profile via matrix multiplication
        max_capacity = np.max(network_profile)
        required_capacity = max(0, max_capacity / infra_obj.cap2act - infra_obj.stock_initial)
        investments_infra[infra_obj.idx] = required_capacity

    tech_stock_infra = tech_stock_exist_infra + investments_infra

    # Determine stocks and investments of buffers
    coord_buffer = np.array([tech.dispatch == 'Gas buffer' for tech in tech_entities])
    nGb = np.sum(coord_buffer)
    if nGb > 0:

        # Determine what’s the previous and new stocks
        techstock_buffer = tech_stock[coord_buffer]
        techstock_exist_buffer = techstock_exist[coord_buffer]

        # To avoid unnecessary decommissioning and negative investments:
        techstock_buffer = np.maximum(techstock_buffer, techstock_exist_buffer)
        investments_buffer = techstock_buffer - techstock_exist_buffer

        # Save the definitive buffers variables
        tech_stock[coord_buffer] = techstock_buffer
        investments[coord_buffer] = investments_buffer

    return tech_stock_infra, investments_infra, tech_stock, investments
