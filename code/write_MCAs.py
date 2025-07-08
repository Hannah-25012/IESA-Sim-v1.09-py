import pandas as pd

def write_MCAs(activities, technologies, agents, writer):
    # Extract Parameters
    periods = activities['periods']
    tech_balancers = technologies['balancers']['ids']
    activity_per_tech = technologies['balancers']['activities']
    tech_names = technologies['balancers']['names']
    tech_sector = technologies['balancers']['sectors']
    tech_subsector = technologies['balancers']['subsectors']
    tech_units = technologies['balancers']['units']
    multi_criteria_performance_tech = technologies['balancers']['mca']['matrix']
    multi_criteria_categories = agents['criteria']['categories']
    
    # Sheet name
    sheet_name = 'Technology_MCAs'
    
    # Build the cell to write
    nP = len(periods)
    nTb = len(tech_balancers)
    nMC = len(multi_criteria_categories)
    
    # Create the header row
    C = []
    header = ['technology', 'name', 'sector', 'subsector', 'main activity', 'units', 'mca category']
    for iP in range(nP):
        header.append(str(periods[iP]))
    C.append(header)
    
    # For each technology report the LCOPs
    for iTb in range(nTb):
        for iMC in range(nMC):
            row = []
            row.append(tech_balancers[iTb])
            row.append(tech_names[iTb])
            row.append(tech_sector[iTb])
            row.append(tech_subsector[iTb])
            row.append(activity_per_tech[iTb])
            row.append(tech_units[iTb])
            row.append(multi_criteria_categories[iMC])
            # Append performance values for all periods.
            # In Matlab, this was done by permuting the 1x1xnP array.
            # Here we assume multi_criteria_performance_tech[iTb][iMC] is a list (or array) of length nP.
            row.extend(multi_criteria_performance_tech[iTb][iMC])
            C.append(row)
    
    # Write the excel sheet (similar to xlswrite in Matlab)
    df = pd.DataFrame(C[1:], columns=C[0])


    df.to_excel(writer, sheet_name=sheet_name, header=False, index=False)
