import pandas as pd

def write_hourly_power_prices(activities, output_name):

    # Extract parameters
    periods = activities['periods']
    activities_elec = activities['electricity']['names']
    activities_elec_coord = activities['electricity']['coords']
    # Note: prices_hourly is extracted using the given coordinates for columns
    prices_hourly = activities['prices']['hourly'][:, activities_elec_coord, :]

    # Sheet name
    sheet_name = 'Hourly_Power_Prices_EURpMWh'

    # Build the cell to write
    nP = len(periods)
    nAk = len(activities_elec)
    nH = prices_hourly.shape[0]  # Number of hours is assumed to be the first dimension
    num_cols = nP * nAk + 1
    num_rows = nH + 2
    c = [[None for _ in range(num_cols)] for _ in range(num_rows)]

    # Fix the headers
    c[1][0] = 'Hour'
    col_counter = 0
    for iAk in range(nAk):
        for i_p in range(nP):
            col_counter += 1
            c[0][col_counter] = activities_elec[iAk]   # First header row: activity name
            c[1][col_counter] = str(periods[i_p])         # Second header row: period as string

    # Fix the content
    col_counter = 0
    # Fill the first column (from row 3 onward) with hour numbers 1 to n_h
    for i in range(nH):
        c[i + 2][0] = i + 1
    for iAk in range(nAk):
        for iP in range(nP):
            col_counter += 1
            # Extract the column vector for the current activity and period, multiply by 3.6
            col_data = prices_hourly[:, iAk, iP] * 3.6
            for i in range(nH):
                c[i + 2][col_counter] = col_data[i]

    # Write the Excel sheet
    # this approach makes sure that the sheet is closed after writing
    df = pd.DataFrame(c)

    df.to_excel(output_name, sheet_name=sheet_name, header=False, index=False)
