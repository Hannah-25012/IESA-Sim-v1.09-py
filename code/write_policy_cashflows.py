import pandas as pd

def write_policy_cashflows(types, activities, results, writer):
    # Extract parameters
    policy_cashflows_categories = types['policy_cashflows_categories']
    periods = activities['periods']
    policy_cashflows = results['policy_cashflows']

    # Sheet name
    sheet_name = 'Policy_cashflows'

    # Build the cell to write
    nPCC = len(policy_cashflows_categories)
    nP = len(periods)
    # Create a 2D list (cell array) with dimensions (n_pcc+1) x (n_p+1)
    c = [['' for _ in range(nP + 1)] for _ in range(nPCC + 1)]
    c[0][0] = 'Cashflows in B€/y'
    # Fill the first row with periods (converted to cells)
    for j in range(nP):
        c[0][j + 1] = periods[j]
    # Fill the first column with policy_cashflows_categories
    for i in range(nPCC):
        c[i + 1][0] = policy_cashflows_categories[i]
    # Fill the remaining cells with policy_cashflows divided by 1000
    for i in range(nPCC):
        for j in range(nP):
            c[i + 1][j + 1] = policy_cashflows[i][j] / 1000

    # Write the excel sheet
    df = pd.DataFrame(c)
    df.to_excel(writer, sheet_name=sheet_name, header=False, index=False)
