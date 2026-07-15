# File to write the projected LCOPs used for the investment decisions
import pandas as pd

def write_LCOPs(activities, technologies, writer):

    # Extract Parameters
    periods = activities.periods
    tech_entities = technologies.balancers.entities
    categories_LCOPs = technologies.balancers.lcops.categories
    tech_LCOPs_matrix = technologies.balancers.lcops.matrix

    # Sheet name
    sheet_name = 'Technology_LCOPs'

    # Build the cell to write
    nP = len(periods)
    nTb = len(tech_entities)
    nLT = len(categories_LCOPs)
    C = [["" for _ in range(nP + 7)] for _ in range(nTb * nLT + 1)] # The dimensions are (nTb*nLT+1) x (nP+7)
    C[0][0] = 'Technology' # Fill header row
    C[0][1] = 'Name'
    C[0][2] = 'Sector'
    C[0][3] = 'Subsector'
    C[0][4] = 'Main Activity'
    C[0][5] = 'Units'
    C[0][6] = 'LCOP category'

    for iP in range(nP):
        C[0][7 + iP] = str(periods[iP])

    # For each technology, report the LCOPs
    i_row = 0
    for tech in tech_entities:

        # For each LCOPs category, report the value
        for iLT in range(nLT):

            # Advance one row
            i_row += 1

            # Store the values in the cell array
            C[i_row][0] = tech.id
            C[i_row][1] = tech.name
            C[i_row][2] = tech.sector
            C[i_row][3] = tech.subsector
            C[i_row][4] = tech.activity_name
            C[i_row][5] = tech.unit
            C[i_row][6] = categories_LCOPs[iLT]

            # Fill LCOP values for each period.
            for iP in range(nP): # This replicates MATLAB's: C(iRow,8:end) = num2cell(permute(tech_LCOPs_matrix(iTb,iLT,:), [1, 3, 2]));
                C[i_row][7 + iP] = tech_LCOPs_matrix[tech.idx][iLT][iP]

    # Write the Excel sheet using pandas
    df = pd.DataFrame(C)
    df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
