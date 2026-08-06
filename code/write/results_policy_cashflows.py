# File to obtain the policy cashflow balance
import numpy as np

def results_policy_cashflows(dimensions, parameters, types, activities, technologies, policies, results):

    # Extract parameters
    nP = dimensions['nP']
    gov_dr = parameters.gov_dr
    policy_cashflows_categories = types.policy_cashflows_categories
    activity_entities = activities.entities
    tech_entities = technologies.balancers.entities
    activity_balances = technologies.balancers.activity_balances
    inv_cost = technologies.balancers.costs.investments
    vom_cost = technologies.balancers.costs.voms
    tech_use = technologies.balancers.use.yearly
    investments = technologies.balancers.investments
    taxes_activities = policies.taxes.activities
    taxes_values = policies.taxes.values
    feedin_activities = policies.feedins.activities
    feedin_values = policies.feedins.values
    subsidy_activities = policies.subsidies.activities
    subsidy_values = policies.subsidies.values
    policy_cashflows = results.policy_cashflows

    # Activity-name -> entity/index lookups, resolved once instead of
    # re-scanning name lists inside the technology loops below.
    activity_by_name = {a.name: a for a in activity_entities}
    taxes_idx_by_name = {name: i for i, name in enumerate(taxes_activities)}
    subsidy_idx_by_name = {name: i for i, name in enumerate(subsidy_activities)}

    # 1) Calculate EUAs========================================================
    # Check the EUA balance of all technologies
    iPc = np.array([cat == 'EUA' for cat in policy_cashflows_categories])
    eua_activity = activity_by_name['CO2 Air ETS']
    coord_national_ets = np.array([tech.subsector == 'National ETS' for tech in tech_entities])
    tech_eua_balance = activity_balances[:, eua_activity.idx].copy() # Select the column corresponding to 'CO2 Air ETS'
    # 'CO2 Air ETS' is an emission-type activity, now stored positive-for-
    # emitters (IESA-Opt convention) - negate this local copy back to the old
    # (negative-for-emitters) convention so eua_balance/eua_cashflow below
    # come out numerically identical to before.
    tech_eua_balance = -tech_eua_balance
    tech_eua_balance[coord_national_ets] = 0 # Set to zero those technologies that are of 'National ETS'
    eua_balance = np.sum(tech_use * tech_eua_balance[:, None], axis=0) # Broadcast the balance to all periods and multiply elementwise with tech use then sum across technologies

    # Multiply by the EUA price
    eua_cashflow = eua_balance * vom_cost[coord_national_ets, :]
    policy_cashflows[iPc, :] = eua_cashflow

    # 2) Calculate all the taxes ==============================================
    iPc = np.array([cat == 'Taxes' for cat in policy_cashflows_categories])
    taxes_cashflow = np.zeros(nP)
    for act in activity_entities:

        # Check if the activity is subject to taxes
        if act.name in taxes_idx_by_name:
            tax_index = taxes_idx_by_name[act.name]

            # Check how much of it is being consumed. Emission-type columns
            # are now stored positive-for-emitters (IESA-Opt convention) -
            # negate this local copy back to the old (negative-for-emitters)
            # convention first so the "keep only negative" clip below still
            # selects the same technologies/amounts as before.
            tech_act_balance = activity_balances[:, act.idx].copy()
            if act.is_emission:
                tech_act_balance = -tech_act_balance
            tech_act_balance[tech_act_balance > 0] = 0
            act_balance = np.sum(tech_use * (tech_act_balance.reshape(-1, 1) @ np.ones((1, nP))), axis=0)

            # Identify the tax value
            tax_value = taxes_values[tax_index, :]

            # Quantify the cashflow
            taxes_cashflow = taxes_cashflow + tax_value * act_balance

    policy_cashflows[iPc, :] = taxes_cashflow

    # 3) Calculate all the feed-in subsidies ==================================
    iPc = np.array([cat == 'Feed-In subsidies' for cat in policy_cashflows_categories])
    feedin_activity_objs = [activity_by_name.get(name) for name in feedin_activities]
    feedin_cashflow = np.zeros(nP)
    for tech in tech_entities:
        iTb = tech.idx

        # Check if it is eligible for a feedin subsidy
        if tech.feedin_subject == 1:

            # Check if there is installed capacity in any year
            if np.sum(tech_use[iTb, :]) > 0:

                # Check if the technology is producing something subject to feed in subsidy
                for iAf, act in enumerate(feedin_activity_objs):
                    if act is not None:

                        # Find balance
                        tech_act_balance = activity_balances[iTb, act.idx]
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
    subsidy_cashflow = np.zeros(nP)
    for tech in tech_entities:
        iTb = tech.idx

        # Check if tech is subject to investment subsidy
        if tech.subsidy_subject == 1:

            # Check if there were investments in any year
            if np.sum(investments[iTb, :]) > 0:

                # Check which is the main activity
                tech_act = tech.activity

                # Check if the activity is subject to investment subsidy
                if tech_act is not None and tech_act.name in subsidy_idx_by_name:
                    iAs_index = subsidy_idx_by_name[tech_act.name]

                    # Quantify the overnight investment costs
                    inv_tech = investments[iTb, :] * inv_cost[iTb, :]

                    # Depreciate the subsidy costs accordingly with government disc. rate
                    tech_lifetime = tech.lifetime
                    annuity_fact = gov_dr / (1 - (1 + gov_dr) ** (-tech_lifetime))
                    dep_inv_tech = annuity_fact * inv_tech

                    # Expand the capital costs for the economic lifetime
                    xP = int(np.ceil(tech_lifetime / 5))
                    for iP in range(nP):
                        validity = min(nP, iP + xP + 1)
                        subsidy_cashflow[iP:validity] += (dep_inv_tech[iP] * subsidy_values[iAs_index, iP] / 100)

    policy_cashflows[iPc, :] = subsidy_cashflow

    # Save variables
    results.policy_cashflows = policy_cashflows

    return results
