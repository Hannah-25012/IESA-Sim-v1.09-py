import numpy as np

def results_policy_cashflows(dimensions, parameters, types, activities, technologies, policies, results):
    # Extract parameters
    nP = dimensions['nP']
    nA = dimensions['nA']
    nTb = dimensions['nTb']
    gov_dr = parameters['gov_dr']
    policy_cashflows_categories = types['policy_cashflows_categories']
    actitivites_names = activities['names']
    activityper_tech = technologies['balancers']['activities']
    activity_balances = technologies['balancers']['activity_balances']
    tech_subsector = technologies['balancers']['subsectors']
    inv_cost = technologies['balancers']['costs']['investments']
    vom_cost = technologies['balancers']['costs']['voms']
    ec_lifetime = technologies['balancers']['costs']['lifetimes']
    subsidy_subject = technologies['balancers']['policies']['subsidy_subject']
    feedin_subject = technologies['balancers']['policies']['feedin_subject']
    tech_use = technologies['balancers']['use']['yearly']
    investments = technologies['balancers']['investments']
    taxes_activities = policies['taxes']['activities']
    taxes_values = policies['taxes']['values']
    feedin_activities = policies['feedins']['activities']
    feedin_values = policies['feedins']['values']
    subsidy_activities = policies['subsidies']['activities']
    subsidy_values = policies['subsidies']['values']
    policy_cashflows = results['policy_cashflows']

    # 1) Calculate EUAs========================================================
    # Check the EUA balance of all technologies
    iPc = np.array([cat == 'EUA' for cat in policy_cashflows_categories])
    iAeua = np.array([act == 'CO2 Air ETS' for act in actitivites_names])
    iTeua = np.array([subsec == 'National ETS' for subsec in tech_subsector])
    # Select all rows and the column(s) corresponding to 'CO2 Air ETS'
    tech_eua_balance = activity_balances[:, iAeua]
    # Set to zero those technologies that are of 'National ETS'
    tech_eua_balance[iTeua, :] = 0
    # Replicate the balance to all periods and multiply elementwise with tech use then sum across technologies
    eua_balance = np.sum(tech_use * (tech_eua_balance @ np.ones((tech_eua_balance.shape[1], nP))), axis=0)
    # Multiply by the EUA price (selecting the matching technology row(s) in vom_cost)
    eua_cashflow = eua_balance * vom_cost[iTeua, :]
    policy_cashflows[iPc, :] = eua_cashflow

    # 2) Calculate all the taxes ==============================================
    iPc = np.array([cat == 'Taxes' for cat in policy_cashflows_categories])
    taxes_cashflow = np.zeros(nP)
    for iA in range(nA):
        # Check if the activity is subject to taxes
        if actitivites_names[iA] in taxes_activities:
            tax_index = taxes_activities.index(actitivites_names[iA])
            # Check how much of it is being consumed
            tech_act_balance = activity_balances[:, iA].copy()
            tech_act_balance[tech_act_balance > 0] = 0
            act_balance = np.sum(tech_use * (tech_act_balance.reshape(-1, 1) @ np.ones((1, nP))), axis=0)
            # Identify the tax value
            tax_value = taxes_values[tax_index, :]
            # Quantify the cashflow
            taxes_cashflow = taxes_cashflow + tax_value * act_balance
    policy_cashflows[iPc, :] = taxes_cashflow

    # 3) Calculate all the feed-in subsidies ==================================
    iPc = np.array([cat == 'Feed-In subsidies' for cat in policy_cashflows_categories])
    nAf = len(feedin_activities)
    feedin_cashflow = np.zeros(nP)
    for iTb in range(nTb):
        # Check if it is eligible for a feedin subsidy
        if feedin_subject[iTb] == 1:
            # Check if there is installed capacity in any year
            if np.sum(tech_use[iTb, :]) > 0:
                # Check if the technology is producing something subject to feed in subsidy
                for iAf in range(nAf):
                    if feedin_activities[iAf] in actitivites_names:
                        iA = actitivites_names.index(feedin_activities[iAf])
                        # Find balance
                        tech_act_balance = activity_balances[iTb, iA]
                        if tech_act_balance < 0:
                            tech_act_balance = 0
                        act_balance = tech_use[iTb, :] * tech_act_balance
                        # Find if it generated something
                        if np.any(act_balance):
                            # Find the subsidy amount
                            feedin_value = feedin_values[iAf, :]
                            # Quantify the cashflow
                            feedin_cashflow = feedin_cashflow + feedin_value * act_balance
    policy_cashflows[iPc, :] = feedin_cashflow

    # Calculate and depreciate all investment subsidies
    iPc = np.array([cat == 'Investment subsidies' for cat in policy_cashflows_categories])
    nAs = len(subsidy_activities)
    subsidy_cashflow = np.zeros(nP)
    for iTb in range(nTb):
        # Check if tech is subject to investment subsidy
        if subsidy_subject[iTb] == 1:
            # Check if there were investments in any year
            if np.sum(investments[iTb, :]) > 0:
                # Check which is the main activity
                tech_act = activityper_tech[iTb]
                # Check if the activity is subject to investment subsidy
                if tech_act in subsidy_activities:
                    iAs_index = subsidy_activities.index(tech_act)
                    # Quantify the overnight investment costs
                    inv_tech = investments[iTb, :] * inv_cost[iTb, :]
                    # Depreciate the subsidy costs accordingly with government disc. rate
                    tech_lifetime = ec_lifetime[iTb]
                    annuity_fact = gov_dr / (1 - (1 + gov_dr) ** (-tech_lifetime))
                    dep_inv_tech = annuity_fact * inv_tech
                    # Expand the capital costs for the economic lifetime
                    xP = int(np.ceil(tech_lifetime / 5))
                    for iP in range(nP):
                        validity = slice(iP, min(nP, iP + xP))
                        subsidy_cashflow[validity] = subsidy_cashflow[validity] + dep_inv_tech[iP] * subsidy_values[iAs_index, iP] / 100
    policy_cashflows[iPc, :] = subsidy_cashflow

    # Save variables
    results['policy_cashflows'] = policy_cashflows
    return results
