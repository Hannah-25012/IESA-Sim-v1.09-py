# File to write the output tech stock sheet of the general output file
import pandas as pd

def write_techUse(activities, technologies, writer):

    # Extract parameters
    periods = activities.periods
    tech_entities = technologies.balancers.entities
    tech_use = technologies.balancers.use.yearly

    # Sheet name
    sheet_name = 'Configuration_Use'

    # Build the cell to write
    nP = len(periods)
    nTb = len(tech_entities)
    C = [[None for _ in range(nP + 6)] for _ in range(nTb + 1)] # Create a list of lists with (nTb+1) rows and (nP+6) columns, initialize with None

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
    for tech in tech_entities:
        i = tech.idx
        C[i + 1][0] = tech.id
        C[i + 1][1] = tech.name
        C[i + 1][2] = tech.sector
        C[i + 1][3] = tech.subsector
        C[i + 1][4] = tech.activity_name
        C[i + 1][5] = tech.unit

        # Assign tech_use values for each period
        for j in range(nP):
            C[i + 1][6 + j] = tech_use[i][j]

    # Write the excel sheet using pandas
    df = pd.DataFrame(C)
    df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
