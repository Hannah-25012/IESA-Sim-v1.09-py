# File to determine the multi-criteria performance of technologies
import numpy as np
from period_balances import balances_for_period, has_period_balances

def invest_multicriteria_performance(dimensions, activities, technologies, agents, iP,
                                      foresight_reference=None):
    """foresight_reference (optional, same object invest_tech_LCOPs.py receives): when given, the emissions
    performance criterion looks ahead the way the LCOP's fuels and emissions terms do. Each technology's
    emission factor (its emission balances, own activity excluded) becomes the discount-weighted average of
    the factor in this period and in every later period inside its remaining lifetime, using that period's
    own coefficient matrix (IESA-Opt-style efficiency learning, period_balances.py). Weights: 1 for this
    period, (1 + r)^-(years ahead) for later ones, r the technology's own discount rate - identical to the
    LCOP. Only technologies whose coefficients change over time are affected; without efficiency-learning
    entries, or without foresight, the criterion is exactly the single-period one."""

    # Extract parameters
    nTb = dimensions['nTb']
    tech_entities = technologies.balancers.entities
    emission_idx = [a.idx for a in activities.entities if a.is_emission]
    tech_LCOPs = technologies.balancers.lcops.values[:, iP]
    multicriteria_performance_tech = technologies.balancers.mca.matrix[:, :, iP]
    multicriteria_categories = agents.criteria.categories

    # Future periods the emission factor may look at (same window rule as invest_tech_LCOPs.py)
    future_balances = None
    if foresight_reference is not None and has_period_balances(technologies):
        periods = activities.periods
        max_period_idx = foresight_reference.energy_prices.shape[1] - 1
        future_idxs_all = [t for t in range(iP + 1, len(periods)) if t <= max_period_idx]
        future_balances = {t: balances_for_period(technologies, t) for t in future_idxs_all}
        discount_rate = technologies.balancers.costs.discount_rate

    # Identify multi-criteria coordinates
    iMC1 = multicriteria_categories.index('Social Attitude')
    iMC2 = multicriteria_categories.index('Emissions performance')
    iMC3 = multicriteria_categories.index('Cost performance')
    iMC4 = multicriteria_categories.index('Complexity')

    # Do a loop for all technologies to calculate their multi-criteria performance
    emissions = np.zeros(nTb) # Preallocate emissions
    for tech in tech_entities:

        # Extract the balance of the technology, zeroing out its own main activity
        technology_balance = np.copy(tech.activity_balances)
        if tech.activity is not None:
            technology_balance[tech.activity.idx] = 0

        # Calculate emissions. No leading minus: technology_balance is now
        # positive-for-emitters (IESA-Opt convention), so the raw sum already
        # is "emitted amount", exactly as the negated sum used to be.
        emissions[tech.idx] = np.sum(technology_balance[emission_idx])

        # Foresight: discount-weighted average of the emission factor over the remaining lifetime
        if future_balances:
            weighted_emissions = emissions[tech.idx]
            weight_total = 1.0
            for t in future_balances:
                if periods[t] > periods[iP] + tech.lifetime:
                    continue
                w = (1 + discount_rate[tech.idx]) ** -(periods[t] - periods[iP])
                balance_t = future_balances[t][tech.idx].copy()
                if tech.activity is not None:
                    balance_t[tech.activity.idx] = 0
                weighted_emissions += w * np.sum(balance_t[emission_idx])
                weight_total += w
            emissions[tech.idx] = weighted_emissions / weight_total

        # Retrieve social perception parameter
        if tech.social_perception == 'Negative':
            multicriteria_performance_tech[tech.idx, iMC1] = 0
        elif tech.social_perception == 'Neutral':
            multicriteria_performance_tech[tech.idx, iMC1] = 0.5
        elif tech.social_perception == 'Positive':
            multicriteria_performance_tech[tech.idx, iMC1] = 1

        # Retrieve perceived complexity parameter
        if tech.complexity == 'Low':
            multicriteria_performance_tech[tech.idx, iMC4] = 1
        elif tech.complexity == 'Med':
            multicriteria_performance_tech[tech.idx, iMC4] = 0.5
        elif tech.complexity == 'High':
            multicriteria_performance_tech[tech.idx, iMC4] = 0

    # Identify emission and LCOP ranges for all technologies with the same main activity
    for activity in activities.entities:

        # Identify competing (non-buffer) technologies for this activity
        competitors = [tech for tech in activity.technologies if not tech.is_buffer]

        if len(competitors) > 0:
            icoord = [tech.idx for tech in competitors]

            # Identify vectors of emissions and LCOPs
            emissions_vec = emissions[icoord]
            LCOP_vec = tech_LCOPs[icoord]

            # Identify maximum and minimum values per technology
            LCOP_min = np.min(LCOP_vec)
            LCOP_max = LCOP_min + 0.5 * abs(LCOP_min) + 1e-6
            emissions_min = np.min(emissions_vec)
            # BUGFIX: this used to be emissions_vec[0] (whichever technology
            # happens to be first in activity.technologies' build order) -
            # not the worst competitor, just an arbitrary one. A true min-max
            # normalization needs the actual maximum among the technologies
            # actually competing for this activity.
            emissions_max = max(np.max(emissions_vec), 0) + 1e-6

            # Calculate the decreasing linear functions from 1 to 0
            # For emissions
            multicriteria_performance_tech[icoord, iMC2] = 1 - (
                (emissions_vec - emissions_min) / (emissions_max - emissions_min)
            )
            # For LCOPs
            multicriteria_performance_tech[icoord, iMC3] = 1 - (
                (LCOP_vec - LCOP_min) / (LCOP_max - LCOP_min)
            )

    # Remove negative results
    multicriteria_performance_tech[:, [iMC2, iMC3]] = np.maximum(
        multicriteria_performance_tech[:, [iMC2, iMC3]], 0
    )

    # Save the multi-criteria performance matrix
    technologies.balancers.mca.matrix[:, :, iP] = multicriteria_performance_tech

    return technologies
