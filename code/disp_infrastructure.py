# File to determine required infrastructure expansion
import numpy as np

def disp_infrastructure(dimensions, activities, technologies, tech_use_hourly, tech_stock, iP):

    # Extract Parameters
    nAi = dimensions['nAi']
    nTi = dimensions['nTi']
    activities_names = activities['names']
    activities_infra = activities['infra']['names']
    activity_balances = technologies['balancers']['activity_balances']
    dispatch_type_tech = technologies['balancers']['dispatch']
    investments = technologies['balancers']['investments'][:, iP]
    if iP == 0:
        techstock_exist = technologies['balancers']['stocks']['initial']
    else:
        techstock_exist = technologies['balancers']['stocks']['evolution'][:, iP-1]
    activity_per_tech_infra = technologies['infra']['activity']
    cap2act_infra = technologies['infra']['cap2acts']
    tech_stock_exist_infra = technologies['infra']['stocks']['initial']
    
    # Identify the network use of activities that have infrastructure
    investments_infra = np.zeros(nTi) # Preallocate
    for iAi in range(nAi):
        coord_act = np.array([act == activities_infra[iAi] for act in activities_names]) # Create boolean arrays for matching strings
        coord_tech = np.array([apt == activities_infra[iAi] for apt in activity_per_tech_infra])
        activity_balances_filtered = -activity_balances[:, coord_act] # Select and invert the corresponding columns from activity_balances
        activity_balances_filtered[activity_balances_filtered > 0] = 0 # Set any positive values to zero
        network_profile = -np.dot(tech_use_hourly, activity_balances_filtered) # Compute the network profile via matrix multiplication
        max_capacity = np.max(network_profile)
        cap_val = cap2act_infra[coord_tech] # For the matching technology, compute the required capacity. It is assumed that coord_tech selects exactly one entry.
        ts_exist_val = tech_stock_exist_infra[coord_tech]
        required_capacity = max(0, max_capacity / cap_val[0] - ts_exist_val[0]) # Take the first element since we assume exactly one match
        idx = np.where(coord_tech)[0][0] 
        investments_infra[idx] = required_capacity # Save the required capacity into investments_infra at the position where coord_tech is True
        
    tech_stock_infra = tech_stock_exist_infra + investments_infra
    
    # Determine stocks and investments of buffers
    coord_buffer = np.array([d == 'Gas buffer' for d in dispatch_type_tech])
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
