# File to write the investments sheet of the general output file
import pandas as pd

def write_investments(activities, technologies, writer):

    # Extract Parameters
    periods = activities.periods
    tech_entities = technologies.balancers.entities
    investments = technologies.balancers.investments

    # Sheet name
    sheet_name = 'Investments_bal'

    # Build the cell to write
    nP = len(periods)
    nTb = len(tech_entities)
    c = [[None for _ in range(nP + 6)] for _ in range(nTb + 1)] # Create a 2D list (cell array) with dimensions (ntb+1) x (np_+6)
    c[0][0] = 'Technology'
    c[0][1] = 'Name'
    c[0][2] = 'Sector'
    c[0][3] = 'Subsector'
    c[0][4] = 'Main Activity'
    c[0][5] = 'Units'
    for iP in range(nP):
        c[0][6 + iP] = str(periods[iP])
    for tech in tech_entities:
        i = tech.idx
        c[i+1][0] = tech.id
        c[i+1][1] = tech.name
        c[i+1][2] = tech.sector
        c[i+1][3] = tech.subsector
        c[i+1][4] = tech.activity_name
        c[i+1][5] = tech.unit
        for j in range(nP):
            c[i+1][6 + j] = investments[i][j]

    # Write the excel sheet
    df = pd.DataFrame(c)
    df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
