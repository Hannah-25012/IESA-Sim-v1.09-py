# File to determine the LCOPs of technologies
import numpy as np
from period_balances import balances_for_period, has_period_balances

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
    # UPDATE: the lookahead now also covers FOM, the VOM base and the emission tax inside VOM (recurring
    # costs whose future-period entries exist). Investment alone stays at the construction-period value.

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
        # IESA-Opt-style efficiency learning (period_balances.py): the coefficient
        # matrix of each FUTURE period, so the lookahead below prices a technology's
        # own future fuel use and emissions with that period's improved coefficients
        # instead of today's. None when the database has no efficiency entries - the
        # lookahead then uses today's coefficients exactly as before.
        future_balances = ({t: balances_for_period(technologies, t) for t in future_idxs_all}
                           if has_period_balances(technologies) else None)
    inv_cost = technologies.balancers.costs.investments[:, iP]
    fom_cost = technologies.balancers.costs.foms[:, iP]
    vom_cost = technologies.balancers.costs.voms[:, iP]
    foms_all = technologies.balancers.costs.foms     # [technology, period]: the foresight lookahead reads future columns
    voms_all = technologies.balancers.costs.voms
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

        # Modify the energy price vector for shedding and flexible technologies.
        # flex_adjusted() swaps the electricity price a shedding / demand-response technology pays for the
        # price of its cheap hours: the average of the cheapest `shedding_guarantee` share of hours (shedding)
        # or a blend of the yearly average and that cheap-hours price (DR shifting). It is a function so the
        # foresight lookahead below can apply the SAME adjustment to each future period's own prices and
        # price ranges, instead of mixing a cheap-hours price for today with plain averages for the future.
        flex_tech = ((tech.shedding_guarantee > 0) or (tech.flexibility_form == 'DR shifting')) and (iP > 0)
        if flex_tech:

            # Find which energy activity is also an electricity activity being used
            iAe_modify = np.array([
                (act.elec_idx is not None) and (energy_balance[i] < 0)
                for i, act in enumerate(energy_activities)
            ])

            def flex_adjusted(prices, ranges, _tech=tech, _mod=iAe_modify):
                out = prices.copy()
                if _tech.shedding_guarantee > 0:
                    rge = np.searchsorted(price_ranges, _tech.shedding_guarantee, side='left')
                    cheap = ranges[rge, _mod]
                    # a range that is not available (NaN: the price source gave no hourly prices for that
                    # activity) falls back to the plain price
                    out[_mod] = np.where(np.isfinite(cheap), cheap, prices[_mod])
                elif _tech.flexibility_form == 'DR shifting':
                    rge = np.searchsorted(price_ranges, _tech.flexibility_nonnegotiable, side='left')
                    cheap = ranges[rge, _mod]
                    cheap = np.where(np.isfinite(cheap), cheap, prices[_mod])
                    out[_mod] = (prices[_mod] * _tech.flexibility_nonnegotiable +
                                 cheap * (1 - _tech.flexibility_nonnegotiable))
                return out

            used_energy_prices = flex_adjusted(energy_prices, energy_prices_ranges)
        else:
            used_energy_prices = energy_prices.copy()

        # Adjust investments with retrofitting
        inv_adjusted = inv_cost[iTb]
        if retrofit_potential[iTb] > 0:
            inv_adjusted = retrofit_cost[iTb]

        # Adjust investments with subsidies
        if tech.subsidy_subject and own_act is not None and own_act.name in subsidy_idx_by_name:
            # The Policies sheet gives 'Overnight investment subsidy' as a positive % (50 = half the overnight cost is paid
            # by the government, and results_policy_cashflows pays it out as +value/100 of the depreciated investment),
            # so the investor's cost is multiplied by (1 - value/100). It used to be (1 + value/100): a 50% MARKUP.
            subsidy_effect = 1 - subsidy_values[subsidy_idx_by_name[own_act.name], iP] / 100
            inv_adjusted *= subsidy_effect

        # Adjust voms with feed-in subsidies and taxes
        vom_adjusted = vom_cost[iTb]

        # Adjust for feed-in subsidies
        # Every activity the technology PRODUCES (balance > 0), the main one included, earns the feed-in once,
        # scaled by its balance coefficient - exactly what results_policy_cashflows pays out.
        # NOTE: this reads tech.activity_balances, NOT technology_balance. technology_balance has the main
        # activity zeroed (it is the 'own activity' excluded from the fuel/emission terms), so looping over it
        # silently dropped the main activity's feed-in altogether: an electrolyser's hydrogen feed-in, a
        # synthetic-fuel plant's fuel feed-in, a generator's electricity feed-in never reached its LCOP.
        # (The earlier "subtracted twice" diagnosis was wrong - the zeroed main activity could not be in the loop.)
        feedin_terms = []   # (row in the feed-in table, activity index) per produced, feed-in-eligible activity
        feedin_now = 0.0
        if tech.feedin_subject:
            for iA_prod in np.where(tech.activity_balances > 0)[0]:
                prod_name = activities.entities[iA_prod].name
                if prod_name in feedin_idx_by_name:
                    feedin_row = feedin_idx_by_name[prod_name]
                    feedin_terms.append((feedin_row, iA_prod))
                    feedin_now += feedin_values[feedin_row, iP] * tech.activity_balances[iA_prod]
            vom_adjusted -= feedin_now

        # Adjust for taxes (emissions only)
        # Identify activities being used. Emission-type balances are stored
        # POSITIVE for emitters (IESA-Opt convention, see energy_balance /
        # activity_balances) - the sign here and the leading minus below are
        # the mirror image of the pre-flip `< 0` selector + unsigned product,
        # chosen so taxes_effect (and thus vom_adjusted) is unchanged.
        activity_balance_coord = np.where((technology_balance > 0) & iAc)[0]
        tax_terms = []   # (row in the tax table, activity index) per taxed emission, for the foresight lookahead
        tax_now = 0.0    # today's tax cost per unit of activity (positive = cost)
        for iA_fuel in activity_balance_coord:
            fuel_activity = activities.entities[iA_fuel]
            if fuel_activity.name in taxes_idx_by_name:
                tax_row = taxes_idx_by_name[fuel_activity.name]
                taxes_effect = -taxes_values[tax_row, iP] * technology_balance[iA_fuel]
                vom_adjusted -= taxes_effect
                tax_terms.append((tax_row, iA_fuel))
                tax_now -= taxes_effect

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
            weighted_feedin = feedin_now
            weighted_tax = tax_now
            weighted_fom = fom_cost[iTb]
            weighted_vom_raw = vom_cost[iTb]
            # Hourly price ranges of each future period (None when the price source has none: then a
            # flexible technology's future electricity is priced at the plain yearly price).
            ref_ranges = getattr(foresight_reference, 'energy_price_ranges', None)
            weight_total = weight_now
            for t in future_idxs:
                w = (1 + discount_rate[iTb]) ** -(periods[t] - periods[iP])
                if future_balances is None:
                    fuel_consumption_t = fuel_consumption
                    emission_balance_t = technology_balance[iAc]
                else:
                    # Same derivation as today's fuel_consumption / technology_balance,
                    # from period t's coefficients. cogen_share stays today's value.
                    balance_t = future_balances[t][iTb].copy()
                    if own_act is not None:
                        balance_t[own_act.idx] = 0
                    fuel_consumption_t = np.maximum(-balance_t[iAe], 0)
                    emission_balance_t = balance_t[iAc]
                price_t = foresight_reference.energy_prices[:, t]
                if flex_tech and ref_ranges is not None:
                    price_t = flex_adjusted(price_t, ref_ranges[:, :, t])
                future_fuels = (cogen_share * np.sum(fuel_consumption_t * price_t)).item()
                future_emissions = (cogen_share * np.sum(
                    emission_balance_t * foresight_reference.emission_prices[:, t])).item()
                # The feed-in schedule of period t on what the technology produces in period t
                # (e.g. the hydrogen feed-in falling 30 -> 0 M EUR/PJ between 2025 and 2050).
                if feedin_terms:
                    balances_t = tech.activity_balances if future_balances is None else future_balances[t][iTb]
                    weighted_feedin += w * sum(feedin_values[row, t] * balances_t[iA] for row, iA in feedin_terms)
                # FOM and the VOM base are recurring costs paid every year of the technology's life, so
                # they are priced with period t's own cost entries; the emission tax inside VOM likewise
                # uses period t's tax schedule on the emissions the technology has in period t. (The
                # investment is paid once, at construction, so it stays at today's value.)
                weighted_fom += w * foms_all[iTb, t]
                weighted_vom_raw += w * voms_all[iTb, t]
                if tax_terms:
                    bal_t = technology_balance if future_balances is None else balance_t
                    weighted_tax += w * sum(taxes_values[row, t] * bal_t[iA] for row, iA in tax_terms)
                weighted_fuels += w * future_fuels
                weighted_emissions += w * future_emissions
                weight_total += w
            fuels = weighted_fuels / weight_total
            emissions = weighted_emissions / weight_total
            # vom above already carries today's feed-in; swap it for the discount-weighted average
            # over the same lifetime window the fuel and emission terms use.
            if feedin_terms:
                vom += cogen_share * (feedin_now - weighted_feedin / weight_total)
            # Same swap for the recurring VOM base, the emission tax and FOM: today's value out, the
            # discount-weighted lifetime average in (investment is deliberately left at today's value).
            vom += cogen_share * ((weighted_vom_raw / weight_total - vom_cost[iTb])
                                  + (weighted_tax / weight_total - tax_now))
            fom = (weighted_fom / weight_total / tech.cap2act).item()
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
