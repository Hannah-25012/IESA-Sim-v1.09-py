# File to write the hourly gas prices
import pandas as pd

def write_hourly_gas_prices(activities, output_name):

    # Extract Parameters
    periods = activities.periods
    gaseous_activities = [a for a in activities.entities if a.is_gaseous]
    activities_gaseous_coord = activities.gaseous.coords
    prices_hourly = activities.prices.hourly[:, activities_gaseous_coord, :]

    # Sheet name
    sheet_name = 'Hourly_Gas_Prices_EURpMWh'

    # Build the cell to write
    nP = len(periods)
    nAg = len(gaseous_activities)
    nH = prices_hourly.shape[0]
    num_rows = nH + 2  # Create an empty cell matrix with dimensions (nH+2) x (nP*nAg+1)
    num_cols = nP * nAg + 1
    c = [[None for _ in range(num_cols)] for _ in range(num_rows)]

    # Fix the headers
    c[1][0] = 'Hour'
    col_index = 1
    for act in gaseous_activities:
        for iP in range(nP):
            c[0][col_index] = act.name
            c[1][col_index] = str(periods[iP])
            col_index += 1

    # Fix the content
    content_col_index = 0
    for h in range(nH): # Fill the first column with hour numbers
        c[h + 2][0] = h + 1

    for iAg, act in enumerate(gaseous_activities):
        for iP in range(nP):
            content_col_index += 1

            # Correct units if it's energy
            if act.is_emission:
                corr_units = 1
            else:
                corr_units = 3.6
            for h in range(nH):
                c[h + 2][content_col_index] = prices_hourly[h, iAg, iP] * corr_units

    # Write the excel sheet
    df = pd.DataFrame(c)
    df.to_excel(output_name, sheet_name=sheet_name, index=False, header=False)
