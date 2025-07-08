import pandas as pd

def write_choices(activities, technologies, agents, writer):
    # File to write the projected LCOPs used for the investment decisions

    # Extract Parameters
    periods = activities['periods']
    tech_balancers = technologies['balancers']['ids']
    activityper_tech = technologies['balancers']['activities']
    tech_names = technologies['balancers']['names']
    tech_sector = technologies['balancers']['sectors']
    tech_subsector = technologies['balancers']['subsectors']
    tech_units = technologies['balancers']['units']
    tech_choices_agent = technologies['balancers']['choices_agent']
    agent_types = agents['types']

    # Sheet name
    sheet_name = 'Agents_Choices'

    # Build the cell to write
    nP = len(periods)
    nTb = len(tech_balancers)
    nAt = len(agent_types)
    # Create a list of lists with dimensions (n_tb*n_at + 1) x (n_p + 7)
    c = [[None for _ in range(nP + 7)] for _ in range(nTb * nAt + 1)]
    c[0][0] = 'Technology'
    c[0][1] = 'Name'
    c[0][2] = 'Sector'
    c[0][3] = 'Subsector'
    c[0][4] = 'Main Activity'
    c[0][5] = 'Units'
    c[0][6] = 'Agent'

    for iP in range(nP):
        c[0][7 + iP] = str(periods[iP])

    # For each technology report the LCOPs
    i_row = 1
    for iTb in range(nTb):
        # For each LCOPs category report the value
        for iAt in range(nAt):
            # Advance one row and store the values in a cell
            c[i_row][0] = tech_balancers[iTb]
            c[i_row][1] = tech_names[iTb]
            c[i_row][2] = tech_sector[iTb]
            c[i_row][3] = tech_subsector[iTb]
            c[i_row][4] = activityper_tech[iTb]
            c[i_row][5] = tech_units[iTb]
            c[i_row][6] = agent_types[iAt]
            # In MATLAB: num2cell(permute(tech_choices_agent(iTb,iAT,:), [1, 3, 2]));
            # Assuming tech_choices_agent is a numpy array of shape (n_tb, n_at, n_p),
            # tech_choices_agent[i_tb, i_at, :] is a 1D array (1 x n_p)
            choices = tech_choices_agent[iTb, iAt, :].tolist()
            c[i_row][7:] = choices
            i_row += 1

    # Write the excel sheet
    df = pd.DataFrame(c)
    df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
