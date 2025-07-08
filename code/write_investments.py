import pandas as pd

# File to write the output tech stock sheet of the general output file

def write_investments(activities, technologies, writer):

    # Extract Parameters
    periods = activities['periods']
    tech_balancers = technologies['balancers']['ids']
    activityper_tech = technologies['balancers']['activities']
    tech_names = technologies['balancers']['names']
    tech_sector = technologies['balancers']['sectors']
    tech_subsector = technologies['balancers']['subsectors']
    tech_units = technologies['balancers']['units']
    investments = technologies['balancers']['investments']

    # Sheet name
    sheet_name = 'Investments_bal'

    # Build the cell to write
    nP = len(periods)
    nTb = len(tech_balancers)
    # Create a 2D list (cell array) with dimensions (ntb+1) x (np_+6)
    c = [[None for _ in range(nP + 6)] for _ in range(nTb + 1)]
    c[0][0] = 'Technology'
    c[0][1] = 'Name'
    c[0][2] = 'Sector'
    c[0][3] = 'Subsector'
    c[0][4] = 'Main Activity'
    c[0][5] = 'Units'
    for iP in range(nP):
        c[0][6 + iP] = str(periods[iP])
    for i in range(nTb):
        c[i+1][0] = tech_balancers[i]
        c[i+1][1] = tech_names[i]
        c[i+1][2] = tech_sector[i]
        c[i+1][3] = tech_subsector[i]
        c[i+1][4] = activityper_tech[i]
        c[i+1][5] = tech_units[i]
        for j in range(nP):
            c[i+1][6 + j] = investments[i][j]

    # Write the excel sheet
    df = pd.DataFrame(c)
    df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)