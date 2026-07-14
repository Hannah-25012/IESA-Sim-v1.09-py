# File to initialize power dispatch
import numpy as np

def disp_initialize_power(dimensions, activities, technologies, profiles, tech_use_hourly, iP):

    # Extract Parameters
    nH = dimensions['nH']
    nA = dimensions['nA']
    nAk = dimensions['nAk']
    activities_names = activities.names
    elec_activities = [a for a in activities.entities if a.is_electricity]
    tech_entities = technologies.balancers.entities
    activity_balances = technologies.balancers.activity_balances
    tech_stock = technologies.balancers.stocks.evolution[:, iP]
    tech_subsector = technologies.balancers.subsectors
    cap2act = technologies.balancers.cap2acts
    vom_cost = technologies.balancers.costs.voms[:, iP]
    shedding_capacity = technologies.balancers.shedding.capacity
    shedding_limits = technologies.balancers.shedding.limits
    shedding_guarantee = technologies.balancers.shedding.guarantee
    flexibility_form = technologies.balancers.flexibility.form
    flexibility_capacity = technologies.balancers.flexibility.capacity
    flexibility_volume = technologies.balancers.flexibility.volume
    flexibility_losses = technologies.balancers.flexibility.losses
    flexibility_nonnegotiable = technologies.balancers.flexibility.nonnegotiable
    flexibility_range_days = technologies.balancers.flexibility.range_days
    profile_type = profiles.types
    hourly_profiles = profiles.shapes
    # Each hourly profile type's shape, resolved once instead of re-scanning
    # profile_type for every technology that uses it.
    profile_shape_by_type = {name: hourly_profiles[:, i] for i, name in enumerate(profile_type)}
    interconnector = profiles.interconnectors
    price_profiles = profiles.prices[:, :, iP]

    # === Obtain figures for power dispatch ===
    # Identify generators, shedders, and interconnectors
    tech_generators_coord = np.array(
        (np.array([subsector == 'Inland generation' for subsector in tech_subsector], dtype=bool) |
        np.array([subsector == 'Generation' for subsector in tech_subsector], dtype=bool) |
        np.array([subsector == 'Undispatched' for subsector in tech_subsector], dtype=bool))
    )
    tech_shedding_coord = np.array((shedding_capacity > 0) & (tech_stock > 0), dtype=bool)
    tech_loadshifts_coord = np.logical_and(np.array(flexibility_form) == 'DR shifting', tech_stock > 0)
    tech_batteries_coord = np.logical_and(np.array(flexibility_form) == 'Storage', tech_stock > 0)
    tech_interconnectors_coord = np.array(tech_subsector) == 'XC Trade'
    nG = np.sum(tech_generators_coord)

    # Preallocate targets
    gen_per_elec = np.zeros((nG, nAk), dtype=bool)
    gen_xc_costs_hourly = np.zeros((nH, nG))
    gen_availability_hourly = np.zeros((nH, nG))
    gen_balance_hourly = np.zeros((nH, nA, nG))
    elec_demand_hourly = np.zeros((nH, nAk))
    elec_cogenerated = np.zeros((nAk, 1))

    # Build the hourly demand profiles of electricity
    for activity in elec_activities:
        iAk = activity.elec_idx

        # Identify technologies that are not generators, shedders, loadshifters, batteries, or interconnectors
        coord_act = np.zeros(nA, dtype=bool)
        coord_act[activity.idx] = True
        coord_tech = (
            (activity_balances[:, coord_act] != 0).any(axis=1).astype(int) *  # Ensures a column vector
            (1 - tech_shedding_coord.astype(int)) *
            (1 - tech_generators_coord.astype(int)) *
            (1 - tech_loadshifts_coord.astype(int)) *
            (1 - tech_batteries_coord.astype(int)) *
            (1 - tech_interconnectors_coord.astype(int))
        ).astype(bool)
        coord_tech = coord_tech.reshape(-1, 1)
        tech_sel = np.where(coord_tech)[0]

        # Sum their electricity demand
        nT = np.sum(coord_tech)
        for iT in range(nT):
            coord_itb = tech_sel[iT]
            elec_demand_hourly[:, iAk] -= tech_use_hourly[:, coord_itb] * activity_balances[coord_itb, coord_act]
            elec_cogenerated[iAk, 0] += max(0, np.sum(tech_use_hourly[:, coord_itb]) * activity_balances[coord_itb, coord_act])

    # Get the generators descriptions
    generator_techs = [t for t in tech_entities if tech_generators_coord[t.idx]]
    gen_vom = vom_cost[tech_generators_coord]
    tech_subsector = np.array(tech_subsector)
    generators_subsector = tech_subsector[tech_generators_coord]
    activity_balance_gen = activity_balances[tech_generators_coord, :]
    generators_activity = tech_stock[tech_generators_coord] * cap2act[tech_generators_coord]

    # Identify the electricity activity and hourly availability profile per
    # generator, all at once instead of a per-generator Python loop.
    gen_elec_idx = np.array([gen.activity.elec_idx for gen in generator_techs])
    gen_act_idx = np.array([gen.activity.idx for gen in generator_techs])
    gen_per_elec[np.arange(nG), gen_elec_idx] = True
    activity_balance_gen[np.arange(nG), gen_act_idx] = 0

    gen_profile_matrix = np.column_stack([profile_shape_by_type[gen.profile] for gen in generator_techs])  # (nH, nG)
    gen_availability_hourly[:, :] = gen_profile_matrix * generators_activity[None, :]

    # Get the hourly variable costs of the generators. This branch is dead in
    # the current data (see NOTE below) but kept as its own small per-generator
    # loop rather than vectorized away, since vectorizing dead code risks
    # hiding it entirely instead of leaving it visibly fixable.
    for iG, gen in enumerate(generator_techs):
        if generators_subsector[iG] == 'Inland generation':  # Interconnectors
            elec_generated = gen.activity.name
            # NOTE: `interconnector` is a plain list, so this comparison never
            # matches (a list is never == a string) - this branch is dead in
            # the current data, preserved as-is rather than silently "fixed".
            if np.sum(interconnector == elec_generated) > 0:
                coord_IC = (interconnector == elec_generated)
                gen_xc_costs_hourly[:, iG] = price_profiles[:, coord_IC]

    # Obtain the hourly activity balances for every generator at once
    gen_balance_hourly[:, :, :] = np.broadcast_to(activity_balance_gen.T, (nH, nA, nG))

    gen_per_elec = gen_per_elec.astype(bool)

    # === Obtain figures for shedding of technologies ===
    shedding_techs = [t for t in tech_entities if tech_shedding_coord[t.idx]]
    nS = len(shedding_techs)

    # Preallocate targets
    shed_min_demand_hourly = np.zeros((nH, nS))
    shed_max_demand_hourly = np.zeros((nH, nS))
    shed_max_volume_hourly = np.zeros((nH, nS))
    shed_min_volume_hourly = np.zeros((nH, nS))
    shed_per_elec = np.zeros((nS, nAk), dtype=bool)
    shed_multiplier = np.zeros((nS, 1))

    # Shed guarantee
    shedding_guarantee_values = shedding_guarantee[tech_shedding_coord]
    shedding_limits_values = shedding_limits[tech_shedding_coord]
    shedding_capacity_values = shedding_capacity[tech_shedding_coord]

    # Obtain the min and max profiles
    tech_stock_shedding = tech_stock[tech_shedding_coord]
    cap2act_shedding = cap2act[tech_shedding_coord]

    for iS, shed in enumerate(shedding_techs):

        # Identify which electricity activities are involved
        elec_balance = 0
        for act in elec_activities:
            act_use = -shed.activity_balances[act.idx]
            if act_use > 0:
                shed_per_elec[iS, act.elec_idx] = True
                elec_balance = act_use

        # Save the shed multiplier
        shed_multiplier[iS, 0] = elec_balance

        # Get the hourly availability profiles of the shedding technologies
        ref_profile = (tech_stock_shedding[iS] * elec_balance *
                    cap2act_shedding[iS] * profile_shape_by_type[shed.profile])
        shed_potential = np.minimum(ref_profile * shedding_limits_values[iS],
                                    tech_stock_shedding[iS] * cap2act_shedding[iS] * shedding_capacity_values[iS])

        # Get the min and max demands and shedding profiles
        shed_max_demand_hourly[:, iS] = ref_profile
        shed_min_demand_hourly[:, iS] = ref_profile - shed_potential
        shed_max_volume_hourly[:, iS] = shed_max_demand_hourly[:, iS] / np.sum(ref_profile)
        shed_min_volume_hourly[:, iS] = shed_min_demand_hourly[:, iS] / np.sum(ref_profile)

    shed_per_elec = shed_per_elec.astype(bool)

    # === Obtain figures for load-shifting technologies ===
    loadshift_techs = [t for t in tech_entities if tech_loadshifts_coord[t.idx]]
    nL = len(loadshift_techs)

    # Preallocate targets
    loadshifts_demand_hourly = np.zeros((nH, nL))
    loadshifts_per_elec = np.zeros((nL, nAk), dtype=bool)
    loadshifts_per_uoa = np.zeros((nL, 1))
    loadshifts_efficiencies = 0
    loadshifts_capacities = 0
    loadshifts_min = 0
    loadshifts_range = 0

    # If there are loadshift technologies
    if nL > 0:
        loadshifts_efficiencies = 1 - flexibility_losses[tech_loadshifts_coord]
        loadshifts_capacities = flexibility_capacity[tech_loadshifts_coord] * tech_stock[tech_loadshifts_coord]
        loadshifts_min = flexibility_nonnegotiable[tech_loadshifts_coord]
        loadshifts_range = flexibility_range_days[tech_loadshifts_coord]

        # Obtain reference profiles
        tech_stock_loadshifts = tech_stock[tech_loadshifts_coord]
        cap2act_loadshifts = cap2act[tech_loadshifts_coord]

        for iL, loadshift in enumerate(loadshift_techs):

            # Identify which electricity activities are involved
            elec_balance = 0
            for act in elec_activities:
                act_use = -loadshift.activity_balances[act.idx]
                if act_use > 0:
                    loadshifts_per_elec[iL, act.elec_idx] = True
                    elec_balance = act_use
                    loadshifts_per_uoa[iL, 0] = act_use

            # Get the demand profile of the load-shifting technologies
            loadshifts_demand_hourly[:, iL] = (tech_stock_loadshifts[iL] * elec_balance *
                                            cap2act_loadshifts[iL] * profile_shape_by_type[loadshift.profile])

    # === Obtain figures for batteries ===
    battery_techs = [t for t in tech_entities if tech_batteries_coord[t.idx]]
    nB = len(battery_techs)

    # Preallocate targets
    bat_per_elec = np.zeros((nB, nAk), dtype=bool)
    bat_efficiency = 0.5
    bat_capacity = 0
    bat_volume = 0
    bat_vom = 0
    bat_stock = 0

    # Only if there are batteries
    if nB > 0:

        # Extract other necessary parameters
        bat_efficiency = 1 - flexibility_losses[tech_batteries_coord]
        bat_capacity = flexibility_capacity[tech_batteries_coord]
        bat_volume = flexibility_volume[tech_batteries_coord]
        bat_vom = vom_cost[tech_batteries_coord]
        bat_stock = tech_stock[tech_batteries_coord]

        # Identify the electricity activity
        bat_per_elec = np.zeros((nB, nAk)) # Preallocate
        for iB, bat in enumerate(battery_techs):

            # Save the link
            bat_per_elec[iB, bat.activity.elec_idx] = 1

        bat_per_elec = bat_per_elec.astype(bool)

    # === Obtain figures for the interconnectors ===
    interconnector_techs = [t for t in tech_entities if tech_interconnectors_coord[t.idx]]
    nI = len(interconnector_techs)

    # Preallocate targets
    xc_efficiencies = np.zeros((nI, 1))
    xc_per_elec = np.zeros((nAk, nAk, nI), dtype=bool)
    xc_demand = np.zeros((nH, nI))
    xc_supply = np.zeros((nH, nI))

    # Interconnector VOM
    xc_vom = vom_cost[tech_interconnectors_coord]

    # Identify the to and froms and the efficiencies
    techStock_interconnectors = tech_stock[tech_interconnectors_coord]
    cap2act_interconnectors = tech_stock[tech_interconnectors_coord] # FIX: both techStock_interconnectors and cap2act_interconnectors are defined in the same way.
    # Suggested fix by copilot: cap2act_interconnectors = cap2act[tech_interconnectors_coord]

    for iI, xc in enumerate(interconnector_techs):

        # Identify the to coordinate
        iAk_to = xc.activity.elec_idx

        # Identify the from coordinate
        from_idx = np.where(xc.activity_balances < 0)[0][0]
        from_activity = activities.entities[from_idx]
        iAk_from = from_activity.elec_idx

        # Save the link
        xc_per_elec[iAk_to, iAk_from, iI] = True

        # Efficiency
        xc_efficiencies[iI, 0] = -1 / xc.activity_balances[from_idx]

        # Get the hourly availability profiles of the XC technologies
        ref_profile = profile_shape_by_type[xc.profile]

        # Supply and demand profiles
        xc_demand[:, iI] = ref_profile * techStock_interconnectors[iI] * cap2act_interconnectors[iI]
        xc_supply[:, iI] = -xc_demand[:, iI] / xc_efficiencies[iI, 0]

    xc_per_elec = xc_per_elec.astype(bool)

    return (gen_vom, gen_balance_hourly, gen_availability_hourly, gen_xc_costs_hourly,
            gen_per_elec, elec_demand_hourly, shedding_guarantee_values, shed_max_volume_hourly,
            shed_min_volume_hourly, shed_per_elec, shed_max_demand_hourly,
            shed_min_demand_hourly, shed_multiplier, loadshifts_efficiencies, loadshifts_capacities,
            loadshifts_min, loadshifts_per_uoa, loadshifts_range, loadshifts_per_elec,
            loadshifts_demand_hourly, bat_efficiency, bat_capacity, bat_volume,
            bat_per_elec, bat_vom, bat_stock,
            xc_efficiencies, xc_vom, xc_per_elec, xc_demand, xc_supply, elec_cogenerated,
            tech_generators_coord, tech_shedding_coord,
            tech_loadshifts_coord, tech_batteries_coord, tech_interconnectors_coord)
