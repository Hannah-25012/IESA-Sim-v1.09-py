import pandas as pd

def write_techUse(activities, technologies, writer):
    # File to write the output tech stock sheet of the general output file

    # Extract parameters
    periods = activities['periods']
    tech_balancers = technologies['balancers']['ids']
    activityper_tech = technologies['balancers']['activities']
    tech_names = technologies['balancers']['names']
    tech_sector = technologies['balancers']['sectors']
    tech_subsector = technologies['balancers']['subsectors']
    tech_units = technologies['balancers']['units']
    tech_use = technologies['balancers']['use']['yearly']

    # Sheet name
    sheet_name = 'Configuration_Use'

    # Build the cell to write
    nP = len(periods)
    nTb = len(tech_balancers)
    # Create a list of lists with (nTb+1) rows and (nP+6) columns, initialize with None
    C = [[None for _ in range(nP + 6)] for _ in range(nTb + 1)]
    
    # Set header row
    C[0][0] = 'Technology'
    C[0][1] = 'Name'
    C[0][2] = 'Sector'
    C[0][3] = 'Subsector'
    C[0][4] = 'Main Activity'
    C[0][5] = 'Units'
    for i in range(nP):
        C[0][6 + i] = str(periods[i])
    
    # Fill in the remaining rows
    for i in range(nTb):
        C[i + 1][0] = tech_balancers[i]
        C[i + 1][1] = tech_names[i]
        C[i + 1][2] = tech_sector[i]
        C[i + 1][3] = tech_subsector[i]
        C[i + 1][4] = activityper_tech[i]
        C[i + 1][5] = tech_units[i]
        # Assign tech_use values for each period
        for j in range(nP):
            C[i + 1][6 + j] = tech_use[i][j]

    # Write the excel sheet using pandas
    df = pd.DataFrame(C)
    df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
