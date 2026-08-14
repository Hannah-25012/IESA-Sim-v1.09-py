# File to determine investments into power technologies
import numpy as np

def invest_power_technologies(dimensions, parameters, activities, technologies, agents, tech_stock_exist, iP):

    # Extract Parameters
    nTb = dimensions['nTb']
    nAT = dimensions['nAT']
    powinv_SPBT_benchmark = parameters.powinv.SPBT_benchmark
    powinv_SPBT_min = parameters.powinv.SPBT_min
    powinv_CR_threshold = parameters.powinv.CR_threshold
    powinv_CR_min = parameters.powinv.CR_min
    powinv_NUF_threshold = parameters.powinv.NUF_threshold
    powinv_NUF_min = parameters.powinv.NUF_min
    tech_entities = technologies.balancers.entities
    inv_cost = technologies.balancers.costs.investments[:, iP]
    tech_stock_deploy = technologies.balancers.stocks.deploy
    tech_stock = technologies.balancers.stocks.evolution[:, iP]
    tech_stock_min = technologies.balancers.stocks.min[:, iP]
    tech_stock_max = technologies.balancers.stocks.max[:, iP]
    generator_capt_rate = technologies.balancers.generators.CR[:, iP-1]
    generator_norm_ut_fact = technologies.balancers.generators.NUF[:, iP-1]
    generator_cash_flow = technologies.balancers.generators.CF[:, iP-1]
    multi_criteria_performance_tech = technologies.balancers.mca.matrix[:, :, iP]
    agent_profiles = agents.profiles
    multi_criteria_categories = agents.criteria.categories
    weights_multi_criteria = agents.criteria.weights
    agents_populations = agents.populations

    # Agent-profile name -> row lookup, resolved once instead of re-scanning
    # agent_profiles for every technology.
    agent_idx_by_name = {name: i for i, name in enumerate(agent_profiles)}

    # Identify the choice per activity and agent type
    iMC3 = np.where(np.array(multi_criteria_categories) == 'Cost performance')[0][0]

    # Loop through all technologies
    new_investments = np.zeros((nTb,1)) # Preallocate new investments
    for tech in tech_entities:
        iTb = tech.idx
        own_act = tech.activity

        # Identify if the main activity of the technology is electricity generation
        if own_act is not None and own_act.is_electricity and inv_cost[iTb] > 0:

            # Activate the volumes accordingly with the quality of their capture rates and capacity factors. If capacity factors are above a threshold and
            # capture rates satisfy a minimum then activate. If capture rates are above a threshold and capacity factors satisfy a minimum then activate.
            #
            # A technology with no operating history yet (zero stock the
            # previous period - e.g. any new-build generator whose
            # stock_initial is 0: Nuclear SMR, Hydrogen Turbines, CCGT wCCS,
            # or a merged database's Offshore Wind/Nuclear) has an undefined
            # (NaN) capture rate and utilization factor - post_generator_indicators.py
            # sets both to NaN when there's no output to measure. Every
            # "NaN >= threshold" is False, so the gate would block it forever:
            # it can't earn the operating history the gate needs without a
            # first investment, and can't get a first investment without that
            # history. Treat an undefined metric as "passes" so a never-built
            # technology gets an initial foothold, after which it has real
            # stock, a real CR/NUF, and is judged normally. The SPBT and
            # multi-criteria screening below still apply (both NaN-guarded),
            # so this is a first look, not a free build - verified to produce
            # plausible clean-firm-capacity buildout trajectories rather than
            # runaway investment.
            cr = generator_capt_rate[iTb]
            nuf = generator_norm_ut_fact[iTb]
            cr_thr_ok = np.isnan(cr) or cr >= powinv_CR_threshold
            cr_min_ok = np.isnan(cr) or cr >= powinv_CR_min
            nuf_thr_ok = np.isnan(nuf) or nuf >= powinv_NUF_threshold
            nuf_min_ok = np.isnan(nuf) or nuf >= powinv_NUF_min
            condition_1 = cr_thr_ok and nuf_min_ok
            condition_2 = nuf_thr_ok and cr_min_ok

            # Batteries are always eligible if cost-effective
            if tech.flexibility_form == 'Storage':
                condition_1 = True

            # Execute the condition
            if condition_1 or condition_2:

                # Calculate the SPBT of the technology. A never-operated
                # technology has zero cash flow, so this would be 0/0 = NaN,
                # which fails the appetite check below (NaN <= min is False)
                # and re-blocks the very first investment - treat "no cash
                # flow yet" as NaN and let it through, consistent with the
                # NaN-eligibility handling above.
                if generator_cash_flow[iTb] != 0:
                    powinv_SPBT_iTb = inv_cost[iTb] * tech_stock[iTb] / generator_cash_flow[iTb]
                else:
                    powinv_SPBT_iTb = np.nan

                # Check if the technology meets the investors' appetite
                if np.isnan(powinv_SPBT_iTb) or powinv_SPBT_iTb <= powinv_SPBT_min:

                    # Determine potential volume ranges for investments accordingly with 1GW to max(1GW,max investment)
                    range_min = 1  # GW
                    # stock_deploy == 0 means "no per-period deploy-rate cap"
                    # (the same reading as invest_investment_potential.py's
                    # own `if tech.stock_deploy > 0` check) - deploy up to the
                    # full remaining headroom rather than being pinned to the
                    # 1 GW floor. stock_deploy is a Sim-only concept IESA-Opt
                    # never provides, so an unmatched merged technology has it
                    # zero-filled; without this a merged generation technology
                    # can only ever add 1 GW per period, far too slow to
                    # replace a retiring fleet (native Sim's own technologies
                    # carry real stock_deploy values and are unaffected).
                    if tech_stock_deploy[iTb] > 0:
                        range_max = max(1, tech_stock_deploy[iTb])
                    else:
                        range_max = max(1, tech_stock_max[iTb] - tech_stock_exist[iTb])
                    potential_range = range_max - range_min

                    # The point in the range will be determined by the actors preferences
                    # A) We recalculate economic performance based on SPBT:
                    # SPBT <= benchmark =1, SPBT >= min = 0
                    multi_criteria_generator = multi_criteria_performance_tech[iTb, :].copy()
                    # No SPBT yet (never operated) -> no economic score to add;
                    # leave the cost-performance criterion at 0 so the first
                    # foothold stays small and driven by the other criteria,
                    # not by a fabricated payback number.
                    if np.isnan(powinv_SPBT_iTb):
                        multi_criteria_generator[iMC3] = 0.0
                    else:
                        powinv_spbt = 1 - (powinv_SPBT_iTb - powinv_SPBT_benchmark) / (powinv_SPBT_min - powinv_SPBT_benchmark)
                        multi_criteria_generator[iMC3] = max(min(1, powinv_spbt), 0)

                    # B) We use the multiCriteria_performance matrix to obtain an indicator from 0 to 1 for each technology and actor, we then multiply
                    # by the range and its population fraction

                    # Identify population vector of agent type based on agent profile.
                    # agent_profile is a Sim-only concept IESA-Opt never provides (see
                    # mod0_load_duckdb._load_activities) - a merged-only electricity
                    # activity (e.g. this project's "Electricity NL - HV"/"- LV") has no
                    # agent profile assigned, so agent_idx_by_name[None] would KeyError.
                    # Same gap, same fix as invest_tech_choices_per_act.py's own
                    # agent_idx_by_name.get(...) lookup: split evenly across agent types
                    # rather than crashing or guessing a single real profile. Every
                    # technology with a real agent_profile_name is unaffected.
                    agent_idx = agent_idx_by_name.get(own_act.agent_profile_name)
                    if agent_idx is not None:
                        population_vector = agents_populations[agent_idx, :]
                    else:
                        population_vector = np.full(nAT, 1.0 / nAT)
                    population_vector = population_vector / np.sum(population_vector)

                    # Quantify the interest per agent and add to the investment counter
                    investment_itb = range_min
                    for iAT in range(nAT):
                        criteria_iat = np.sum(weights_multi_criteria[:, iAT] * multi_criteria_generator) / np.sum(weights_multi_criteria[:, iAT])
                        investment_itb += potential_range * criteria_iat * population_vector[iAT]

                    new_investments[iTb] = investment_itb

            # Ensure volumes satisfy tech_stock min and max constraints
            up_room = tech_stock_max[iTb] - tech_stock_exist[iTb]
            min_room = max(0, (tech_stock_min[iTb] - tech_stock_exist[iTb]))
            new_investments[iTb] = max(min(up_room, new_investments[iTb]), min_room)

    return new_investments
