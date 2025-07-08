import pandas as pd

def write_energy_prices(activities, writer):
    # Extract Parameters
    periods = activities['periods']
    activities_energy = activities['energies']['names']
    energy_prices = activities['energies']['prices']['yearly']

    # Sheet name
    sheet_name = 'energy_prices'

    # Build the cell to write
    nAe = len(activities_energy)
    nP = len(periods)
    c = [[None] * (nP + 1) for _ in range(nAe + 1)]
    c[0][0] = 'Energy'
    for j in range(nP):
        c[0][j + 1] = periods[j]
    for i in range(nAe):
        c[i + 1][0] = activities_energy[i]
    for i in range(nAe):
        for j in range(nP):
            c[i + 1][j + 1] = energy_prices[i][j]

    # Write the excel sheet
    df = pd.DataFrame(c)
    df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)