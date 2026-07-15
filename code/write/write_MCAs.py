# File to write the technology MCAs sheet
import pandas as pd

def write_MCAs(activities, technologies, agents, writer):

    # Extract Parameters
    periods = activities.periods
    tech_entities = technologies.balancers.entities
    multi_criteria_performance_tech = technologies.balancers.mca.matrix
    multi_criteria_categories = agents.criteria.categories

    # Sheet name
    sheet_name = 'Technology_MCAs'

    # Build the cell to write
    nP = len(periods)
    nMC = len(multi_criteria_categories)
    C = [] # Create the header row
    header = ['technology', 'name', 'sector', 'subsector', 'main activity', 'units', 'mca category']
    for iP in range(nP):
        header.append(str(periods[iP]))
    C.append(header)

    # For each technology report the LCOPs
    for tech in tech_entities:
        for iMC in range(nMC):
            row = []
            row.append(tech.id)
            row.append(tech.name)
            row.append(tech.sector)
            row.append(tech.subsector)
            row.append(tech.activity_name)
            row.append(tech.unit)
            row.append(multi_criteria_categories[iMC])
            # Append performance values for all periods.
            # In Matlab, this was done by permuting the 1x1xnP array.
            # Here we assume multi_criteria_performance_tech[iTb][iMC] is a list (or array) of length nP.
            row.extend(multi_criteria_performance_tech[tech.idx][iMC])
            C.append(row)

    # Write the excel sheet (similar to xlswrite in Matlab)
    df = pd.DataFrame(C[1:], columns=C[0])
    df.to_excel(writer, sheet_name=sheet_name, header=False, index=False)
