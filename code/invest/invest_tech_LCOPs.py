# File to determine the LCOPs of technologies
import numpy as np

def invest_tech_LCOPs(dimensions, activities, technologies, policies, retrofit_potential, retrofit_cost, iP):

    # Extract parameters
    nTb = dimensions['nTb']
    nTL = dimensions['nTL']
    tech_entities = technologies.balancers.entities
    energy_activities = [a for a in activities.entities if a.is_energy]
    if iP == 0:
        energy_prices = activities.energies.prices.initialized
        emission_prices = activities.emissions.prices.initialized
    else:
        energy_prices = activities.energies.prices.yearly[:, iP - 1]
        emission_prices = activities.emissions.prices.yearly[:, iP - 1]
        energy_prices_ranges = activities.energies.prices.ranges[:, :, iP - 1]
    price_ranges = activities.energies.prices.price_ranges
    annuity_fact = technologies.balancers.costs.annuity
    inv_cost = technologies.balancers.costs.investments[:, iP]
    fom_cost = technologies.balancers.costs.foms[:, iP]
    vom_cost = technologies.balancers.costs.voms[:, iP]
    subsidy_activities = policies.subsidies.activities
    subsidy_values = np.array(policies.subsidies.values)
    feedin_activities = policies.feedins.activities
    feedin_values = np.array(policies.feedins.values)
    taxes_activities = policies.taxes.activities
    taxes_values = np.array(policies.taxes.values)

    # Activity-name -> period-column lookups, resolved once instead of
    # re-scanning the policy activity lists for every technology.
    subsidy_idx_by_name = {name: i for i, name in enumerate(subsidy_activities)}
    feedin_idx_by_name = {name: i for i, name in enumerate(feedin_activities)}
    taxes_idx_by_name = {name: i for i, name in enumerate(taxes_activities)}

    # Identify energy and emissions coordinates (boolean masks over ALL activities)
    iAc = np.array([a.is_emission for a in activities.entities])
    iAe = np.array([a.is_energy for a in activities.entities])

    # Do a loop on all technologies to obtain their LCOPs
    tech_lcops = np.zeros(nTb) # Preallocate arrays
    tech_lcops_matrix = np.zeros((nTb, nTL))
    for tech in tech_entities:
        iTb = tech.idx
        own_act = tech.activity

        # Modify the vector included for the activity balance
        technology_balance = tech.activity_balances.copy()
        if own_act is not None:
            technology_balance[own_act.idx] = 0
        energy_balance = technology_balance[iAe]

        # Modify the energy price vector for shedding and flexible technologies
        used_energy_prices = energy_prices.copy()
        if ((tech.shedding_guarantee > 0) or (tech.flexibility_form == 'DR shifting')) and (iP > 0):

            # Find which energy activity is also an electricity activity being used
            iAe_modify = np.array([
                (act.elec_idx is not None) and (energy_balance[i] < 0)
                for i, act in enumerate(energy_activities)
            ])

            # Modify energy prices based on flexibility and price ranges
            if tech.shedding_guarantee > 0:
                price_rge = np.searchsorted(price_ranges, tech.shedding_guarantee, side='left')
                used_energy_prices[iAe_modify] = energy_prices_ranges[price_rge, iAe_modify]
            elif tech.flexibility_form == 'DR shifting':
                price_rge = np.searchsorted(price_ranges, tech.flexibility_nonnegotiable, side='left')
                used_energy_prices[iAe_modify] = (
                    energy_prices[iAe_modify] * tech.flexibility_nonnegotiable +
                    energy_prices_ranges[price_rge, iAe_modify] * (1 - tech.flexibility_nonnegotiable)
                )

        # Adjust investments with retrofitting
        inv_adjusted = inv_cost[iTb]
        if retrofit_potential[iTb] > 0:
            inv_adjusted = retrofit_cost[iTb]

        # Adjust investments with subsidies
        if tech.subsidy_subject and own_act is not None and own_act.name in subsidy_idx_by_name:
            subsidy_effect = 1 + subsidy_values[subsidy_idx_by_name[own_act.name], iP] / 100
            inv_adjusted *= subsidy_effect

        # Adjust voms with feed-in subsidies and taxes
        vom_adjusted = vom_cost[iTb]

        # Adjust for feed-in subsidies
        if tech.feedin_subject:

            # First modify the main activity
            if own_act is not None and own_act.name in feedin_idx_by_name:
                feedin_effect = feedin_values[feedin_idx_by_name[own_act.name], iP]
                vom_adjusted -= feedin_effect

            # Then adjust other activities
            activity_balance_coord = np.where(technology_balance > 0)[0]
            for iA_cogen in activity_balance_coord:
                cogen_activity = activities.entities[iA_cogen]
                if cogen_activity.name in feedin_idx_by_name:
                    feedin_effect = feedin_values[feedin_idx_by_name[cogen_activity.name], iP] * technology_balance[iA_cogen]
                    vom_adjusted -= feedin_effect

        # Adjust for taxes (emissions only)
        # Identify activities being used. Emission-type balances are stored
        # POSITIVE for emitters (IESA-Opt convention, see energy_balance /
        # activity_balances) - the sign here and the leading minus below are
        # the mirror image of the pre-flip `< 0` selector + unsigned product,
        # chosen so taxes_effect (and thus vom_adjusted) is unchanged.
        activity_balance_coord = np.where((technology_balance > 0) & iAc)[0]
        for iA_fuel in activity_balance_coord:
            fuel_activity = activities.entities[iA_fuel]
            if fuel_activity.name in taxes_idx_by_name:
                taxes_effect = -taxes_values[taxes_idx_by_name[fuel_activity.name], iP] * technology_balance[iA_fuel]
                vom_adjusted -= taxes_effect

        # Clean fuel consumption for cogeneration cases
        fuel_consumption = -energy_balance
        fuel_consumption[fuel_consumption < 0] = 0
        if own_act is not None and iAe[own_act.idx]:
            cogen_profile = tech.activity_balances[iAe]
            if np.sum(cogen_profile[cogen_profile > 0]) == 0:
                cogen_share = 1
            else:
                cogen_share = tech.activity_balances[own_act.idx] / np.sum(cogen_profile[cogen_profile > 0])
        else:
            cogen_share = 1

        # Calculate the LCOP
        inv = (inv_adjusted * annuity_fact[iTb] / tech.cap2act).item()
        fom = (fom_cost[iTb] / tech.cap2act).item()
        vom = (cogen_share * vom_adjusted).item()
        fuels = (cogen_share * np.sum(fuel_consumption * used_energy_prices)).item()
        # No leading minus: technology_balance[iAc] is now positive-for-emitters
        # (IESA-Opt convention), so the raw product already carries the right sign.
        emissions = (cogen_share * np.sum(technology_balance[iAc] * emission_prices)).item()

        # Save the LCOPs
        tech_lcops_matrix[iTb, :] = [inv, fom, vom, fuels, emissions]
        tech_lcops[iTb] = np.sum(tech_lcops_matrix[iTb, :])

    # Save the variables
    technologies.balancers.lcops.values[:, iP] = tech_lcops
    technologies.balancers.lcops.matrix[:, :, iP] = tech_lcops_matrix

    return technologies
