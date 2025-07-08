import pandas as pd

def write_sectoral_emissions(activities, types, results, writer):
    # File to write sectoral emissions

    # Extract parameters
    periods = activities['periods']
    sectors = types['sectors']
    emissions_sector_pos = results['emissions_sector_pos']
    emissions_sector_neg = results['emissions_sector_neg']

    # Sheet name
    sheet_name = 'sectoral_emissions'

    # Build the cell to write
    nS = len(sectors)
    nP = len(periods)
    # Create a 2D list (cell) with dimensions (2*n_s+1) x (n_p+2)
    c = [[None for _ in range(nP + 2)] for _ in range(2 * nS + 1)]
    c[0][0] = 'Emisions in Mton CO_2 eq'
    # Fill first row: set columns 1 to n_p with period values
    c[0][1:nP+1] = list(periods[:nP])
    
    # Fill even-indexed rows (MATLAB rows 2,4,... => Python indices 1,3,...): set sector names and "Positive"
    for i in range(nS):
        c[2 * i + 1][0] = sectors[i]
        c[2 * i + 1][1] = 'Positive'
    
    # Fill odd-indexed rows (MATLAB rows 3,5,... => Python indices 2,4,...): set sector names and "Negative"
    for i in range(nS):
        c[2 * i + 2][0] = sectors[i]
        c[2 * i + 2][1] = 'Negative'
    
    # Assign positive emissions data to even-indexed rows starting from column 2
    # and negative emissions data to odd-indexed rows starting from column 2
    for i in range(nS):
        c[2 * i + 1][2:] = list(emissions_sector_pos[i])
        c[2 * i + 2][2:] = list(emissions_sector_neg[i])
    
    # Write the excel sheet
    df = pd.DataFrame(c)

    df.to_excel(writer, sheet_name=sheet_name, header=False, index=False)
