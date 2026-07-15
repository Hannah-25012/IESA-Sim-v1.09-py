# File to write the emission prices sheet
import pandas as pd

def write_emission_prices(activities, writer):

    # Extract Parameters
    periods = activities.periods
    emission_activities = [a for a in activities.entities if a.is_emission]
    emission_prices = activities.emissions.prices.yearly

    # Sheet name
    sheet_name = "Emission_prices"

    # Build the cell to write
    nAc = len(emission_activities)
    nP = len(periods)
    C = [[None] * (nP + 1) for _ in range(nAc + 1)]
    C[0][0] = "Energy"
    for j in range(nP):
        C[0][j + 1] = periods[j]
    for act in emission_activities:
        C[act.emission_idx + 1][0] = act.name
    for act in emission_activities:
        i = act.emission_idx
        for j in range(nP):
            C[i + 1][j + 1] = emission_prices[i][j]

    # Write the excel sheet
    df = pd.DataFrame(C)
    df.to_excel(writer, sheet_name=sheet_name, header=False, index=False)
