import pandas as pd

def write_system_emisisons(activities, results, writer):
    # Export Parameters
    periods = activities['periods']
    emissions = results['emissions']

    # Sheet name
    sheet_name = 'System_emissions'

    # Build the cell to write
    nP = len(periods)
    c = [[None] * (nP + 1) for _ in range(2)]
    c[0][0] = 'Emisions in Mton CO_2 eq'
    c[0][1:] = list(periods)
    c[1][0] = 'Total'
    c[1][1:] = list(emissions)

    # Write the excel sheet
    df = pd.DataFrame(c)
    df.to_excel(writer, sheet_name=sheet_name, header=False, index=False)
