import numpy as np

def invest_techstocks_def(dimensions, technologies, tech_stock_original, preliminary_investments, report_yes, iP):

    # Extract parameters
    nP = dimensions['nP']
    nTb = dimensions['nTb']
    tech_balancers = technologies['balancers']['ids']
    tech_categories = technologies['balancers']['categories']
    tech_sector = technologies['balancers']['sectors']
    inv_cost = technologies['balancers']['costs']['investments'][:, iP]
    tech_stock_min = technologies['balancers']['stocks']['min'][:, iP]
    tech_stock_max = technologies['balancers']['stocks']['max'][:, iP]
    decommissionings = technologies['balancers']['decommissionings']

    # Preallocate the existing tech stocks
    tech_stock_original = tech_stock_original.reshape(-1, 1)
    tech_stock = tech_stock_original + preliminary_investments
    tech_stock_new = np.zeros((nTb,1))  # Preallocate
    approved_investments = np.zeros((nTb,1))  # Preallocate
    forced_investments = np.zeros((nTb,1))  # Preallocate
    forced_decommissionings = np.zeros((nTb,1))  # Preallocate

    # Force all technologies to fall within min and max limits
    for iTb in range(nTb):
        # Determine allowed stock
        tech_stock_new[iTb,0] = min(max(tech_stock_min[iTb], tech_stock[iTb]), tech_stock_max[iTb])

        # Approve investments that do not violate the constraint
        approved_investments[iTb,0] = min(preliminary_investments[iTb,0], 
                                        tech_stock_new[iTb,0] - tech_stock_original[iTb,0])

        # Calculate the delta
        delta_stock = tech_stock_new[iTb,0] - tech_stock[iTb]

        # Check the direction of the delta
        if delta_stock > 0:  # If the delta is positive, invest
            forced_investments[iTb,0] = max(0, delta_stock)
        elif delta_stock < 0:  # If the delta is negative, decommission
            forced_decommissionings[iTb,0] = -min(0, delta_stock)

        # Define the final stocks
        tech_stock[iTb,0] = tech_stock_new[iTb,0]

    # Make all primary energy equal to tech stock max (it does not reflect on investments)
    primary_decommissionings = np.zeros((nTb,1))  # Preallocate
    for iTb in range(nTb):
        if 'Primary' in tech_categories[iTb]:
            tech_stock[iTb] = tech_stock_max[iTb]
            primary_decommissionings[iTb] = tech_stock_max[iTb]
        elif 'Emission' in tech_categories[iTb]:
            if inv_cost[iTb] == 0:
                tech_stock[iTb] = tech_stock_max[iTb]
                primary_decommissionings[iTb] = tech_stock_max[iTb]
            if 'Emission' in tech_sector[iTb]:
                tech_stock[iTb] = 5000
                primary_decommissionings[iTb] = 5000

    # Update primary decommissioning for next period
    if iP + 1 < nP:
        decommissionings[:, iP + 1] += primary_decommissionings[:, 0]

    # Report the definitive stocks if requested
    if report_yes:
        print(f"{'Technology':60s}, {'Tech Stock':10s}")
        for itb in range(nTb):
            print(f"{tech_balancers[itb]:60s}, {tech_stock[itb]:10.2f}")

    # Save variables
    technologies['balancers']['stocks']['evolution'][:, iP] = tech_stock.flatten()
    technologies['balancers']['investments'][:, iP] = approved_investments.flatten() + forced_investments.flatten()
    technologies['balancers']['decommissionings'] = decommissionings

    return technologies, forced_decommissionings
