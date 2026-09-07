# File to determine the LCOPs of technologies
import numpy as np

def invest_tech_LCOPs(dimensions, activities, technologies, policies, retrofit_potential, retrofit_cost, iP,
                       foresight_reference=None):
    """foresight_reference (optional): a Struct with .energy_prices [nAe, nP]
    and .emission_prices [nAc, nP] holding a PREVIOUS full pass's own
    realized yearly prices for every period - see main.py's multi-pass
    orchestration. When None (the default, and always the case when the
    `foresight` setting is off), behavior is byte-identical to the original
    single-period LCOP. When provided, the fuels/emissions cost terms below
    are computed as a discount-weighted average over the technology's own
    remaining lifetime instead of a single-period snapshot - see the
    'foresight-aware fuels/emissions' block below for why only those two
    terms change (investment/fom/vom are already period-specific exogenous
    input data, not something "foresight" can improve on)."""

    # Extract parameters
    nTb = dimensions['nTb']
    nTL = dimensions['nTL']
    tech_entities = technologies.balancers.entities
    energy_activities = [a for a in activities.entities if a.is_energy]
    periods = activities.periods
    if iP == 0:
        energy_prices = activities.energies.prices.initialized
        emission_prices = activities.emissions.prices.initialized
    else:
        energy_prices = activities.energies.prices.yearly[:, iP - 1]
        emission_prices = activities.emissions.prices.yearly[:, iP - 1]
        energy_prices_ranges = activities.energies.prices.ranges[:, :, iP - 1]
    price_ranges = activities.energies.prices.price_ranges
    annuity_fact = technologies.balancers.costs.annuity
    discount_rate = technologies.balancers.costs.discount_rate

    # Foresight lookahead: which future period indices (t > iP) fall within
    # a technology's own remaining economic life, real years (not period-
    # index count - period spacing is uneven, e.g. 2022->2025 is 3 years,
    # 2025->2030 is 5) - and only as far as foresight_reference actually has
    # data for (always the full horizon in practice, since it's captured
    # from a completed prior pass, but bounded defensively regardless).
    if foresight_reference is not None:
        max_period_idx = foresight_reference.energy_prices.shape[1] - 1
        future_idxs_all = [t for t in range(iP + 1, len(periods))
                            if t <= max_period_idx]
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
        fuels_now = (cogen_share * np.sum(fuel_consumption * used_energy_prices)).item()
        # No leading minus: technology_balance[iAc] is now positive-for-emitters
        # (IESA-Opt convention), so the raw product already carries the right sign.
        emissions_now = (cogen_share * np.sum(technology_balance[iAc] * emission_prices)).item()

        # Foresight-aware fuels/emissions: a discount-weighted average of
        # this period's own value (computed exactly as above - including
        # today's shedding/flexibility price adjustment, which a prior
        # pass's plain yearly reference doesn't carry) plus each future
        # period's own value from foresight_reference's trajectory, out to
        # the technology's own remaining lifetime. investment/fom/vom are
        # deliberately left untouched above - they're already period-
        # specific exogenous input data (technology_costs, subsidy/feedin/
        # tax schedules), not something foresight can improve on the way it
        # can for fuels/emissions, whose CURRENT-period prices are
        # themselves only a proxy for what the technology will actually pay
        # to operate over its life (see invest_tech_LCOPs.py's own module
        # docstring). Simplification: future periods use the reference's
        # plain yearly price, not a re-derived shedding/flexibility
        # adjustment for that future period - carrying the full hourly
        # price-range state forward per technology would be a much larger
        # change for a marginal accuracy gain here.
        if foresight_reference is not None:
            future_idxs = [t for t in future_idxs_all if periods[t] <= periods[iP] + tech.lifetime]
            weight_now = 1.0
            weighted_fuels = fuels_now
            weighted_emissions = emissions_now
            weight_total = weight_now
            for t in future_idxs:
                w = (1 + discount_rate[iTb]) ** -(periods[t] - periods[iP])
                future_fuels = (cogen_share * np.sum(
                    fuel_consumption * foresight_reference.energy_prices[:, t])).item()
                future_emissions = (cogen_share * np.sum(
                    technology_balance[iAc] * foresight_reference.emission_prices[:, t])).item()
                weighted_fuels += w * future_fuels
                weighted_emissions += w * future_emissions
                weight_total += w
            fuels = weighted_fuels / weight_total
            emissions = weighted_emissions / weight_total
        else:
            fuels = fuels_now
            emissions = emissions_now

        # Save the LCOPs
        tech_lcops_matrix[iTb, :] = [inv, fom, vom, fuels, emissions]
        tech_lcops[iTb] = np.sum(tech_lcops_matrix[iTb, :])

    # Save the variables
    technologies.balancers.lcops.values[:, iP] = tech_lcops
    technologies.balancers.lcops.matrix[:, :, iP] = tech_lcops_matrix

    return technologies
