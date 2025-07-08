import pandas as pd

def write_emission_prices(activities, writer):
    # File to write the output tech stock sheet of the general output file

    # Extract Parameters
    periods = activities['periods']
    activities_emission = activities['emissions']['names']
    emission_prices = activities['emissions']['prices']['yearly']

    # Sheet name
    sheet_name = "Emission_prices"

    # Build the cell to write
    nAc = len(activities_emission)
    nP = len(periods)
    C = [[None] * (nP + 1) for _ in range(nAc + 1)]
    C[0][0] = "Energy"
    for j in range(nP):
        C[0][j + 1] = periods[j]
    for i in range(nAc):
        C[i + 1][0] = activities_emission[i]
    for i in range(nAc):
        for j in range(nP):
            C[i + 1][j + 1] = emission_prices[i][j]

    # Write the excel sheet
    df = pd.DataFrame(C)
    df.to_excel(writer, sheet_name=sheet_name, header=False, index=False)
