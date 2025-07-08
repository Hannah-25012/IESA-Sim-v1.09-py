import numpy as np
import pandas as pd

def write_sector_balance(types, activities, technologies, writer):
    # Extract Parameters
    energy_labels = types['energy']['labels']
    periods = activities['periods']
    activity_label = activities['labels']
    tech_categories = technologies['balancers']['categories']
    tech_sector = technologies['balancers']['sectors']
    activity_balances = technologies['balancers']['activity_balances']
    tech_use = technologies['balancers']['use']['yearly']
    
    # Sheet name
    sheet_name = 'Sectoral_balance'
    
    # Modify the consumption and production balances
    consumption_balance = activity_balances.copy()
    consumption_balance[activity_balances > 0] = 0
    production_balance = activity_balances.copy()
    production_balance[activity_balances < 0] = 0
    
    # Do the operation for all sectors and all labels
    coord_categories = np.logical_not(np.isin(tech_categories, ['Primary', 'Emission', 'Exports']))
    sectors = np.unique(np.array(tech_sector)[coord_categories])
    nP = tech_use.shape[1]
    nS = len(sectors)
    nEL = len(energy_labels)
    
    # Build the cell to write (a list of lists)
    rows = 2 + nS * nEL
    cols = 2 + 2 * nP
    C = [[None for _ in range(cols)] for _ in range(rows)]
    
    # Header row 1 (Matlab row 1 -> Python index 0)
    C[0][2] = "Consumption"
    C[0][2 + nP] = "Production"
    
    # Header row 2 (Matlab row 2 -> Python index 1)
    C[1][0] = "Sector"
    C[1][1] = "Carrier"
    C[1][2:2 + nP] = list(periods[:nP])
    C[1][nP + 2:2 * nP + 2] = list(periods[:nP])
    
    iR = 2  # Starting row for data (Matlab row 3 -> Python index 2)
    for s in sectors:
        for el in energy_labels:
            # Identify the coordinates
            tech_coord = np.logical_and(coord_categories, np.array(tech_sector) == s)
            act_coord = np.array(activity_label) == el
            
            # Sum the consumption and production of all technologies in the sector and label
            cons_sum = np.sum(consumption_balance[tech_coord][:, act_coord], axis=1)
            consumption = np.sum(tech_use[tech_coord, :] * cons_sum[:, None], axis=0)
            prod_sum = np.sum(production_balance[tech_coord][:, act_coord], axis=1)
            production = np.sum(tech_use[tech_coord, :] * prod_sum[:, None], axis=0)
            
            # Store the data in the cell
            C[iR][0] = s
            C[iR][1] = el
            C[iR][2:2 + nP] = consumption.tolist()
            C[iR][nP + 2:2 * nP + 2] = production.tolist()
            
            # Advance
            iR += 1
    
    # Write the excel sheet
    df = pd.DataFrame(C)
    df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
