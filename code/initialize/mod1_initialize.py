# File to intitialize model dimensions and variables
import numpy as np
from initialize_prices import initialize_prices
from datastructures import Struct

def mod1_initialize(settings, types, activities, technologies, agents, policies):

    # Extract inputs
    energy_labels = types.energy.labels
    energyPrice_init = types.energy.price_init
    periods = activities.periods
    activity_names = activities.names
    activity_type_act = activities.types
    activity_resolution = activities.resolution
    activity_agent = activities.agents
    activity_label = activities.labels
    tech_balancers = technologies.balancers.ids
    tech_categories = technologies.balancers.categories
    ec_lifetime = technologies.balancers.costs.lifetimes
    activity_per_tech = technologies.balancers.activities
    tech_wacc = technologies.balancers.costs.wacc
    vom_cost = technologies.balancers.costs.voms
    vom_cost_init = vom_cost[:, 0]
    flexibility_range = technologies.balancers.flexibility.range
    tech_stock_exist = technologies.balancers.stocks.initial
    tech_stock_dec = technologies.balancers.stocks.dec_planned
    tech_infra = technologies.infra.ids
    activity_per_tech_infra = technologies.infra.activity
    ec_lifetime_infra = technologies.infra.costs.lifetimes
    agent_profiles = agents.profiles
    agents_dr = agents.rates
    agent_types = agents.types
    multi_criteria_categories = agents.criteria.categories
    taxes_activities = policies.taxes.activities
    taxes_values = np.array(policies.taxes.values)
    sectors = types.sectors

    # Hardcoded model parameters
    nH = 8760
    nDy = 365
    nHd = 24
    nIp_min = 3
    nIp_max = 21
    nRp = 21  # Number of price ranges for shedding and shifting technologies

    # === Identify activity types per activity ===
    print('--Allocating sets')
    activities_driver = [name for name, typ in zip(activity_names, activity_type_act) if typ == 'Driver']
    activities_energy = [name for name, typ in zip(activity_names, activity_type_act) if typ in ['Energy', 'Fix Energy']]
    # 'EmissionReport' is IESA-Opt's report-only type for energy-CO2 activities
    # a merged database carries unchanged (see entities.py:is_emission). Native
    # Sim has none, so this only ever matters for a merged run.
    activities_emission = [name for name, typ in zip(activity_names, activity_type_act) if typ in ('Emission', 'EmissionReport')]

    # === Identify the electricity activities ===
    activities_elec_coord = np.array(activity_label) == 'Electricity'
    activities_elec = np.array(activity_names)[activities_elec_coord]

    # === Identify the gaseous network activities ===
    activities_gaseous_coord = np.array(activity_resolution) == 'daily'
    activities_gaseous = np.array(activity_names)[activities_gaseous_coord]

    # === Identify activity types per technology ===
    nTb = len(tech_balancers)
    nTi = len(tech_infra)
    technologies_driver = np.zeros(nTb, dtype=bool)
    technologies_energy = np.zeros(nTb, dtype=bool)
    technologies_emission = np.zeros(nTb, dtype=bool)
    for iTb in range(nTb):
        tech_activity = activity_per_tech[iTb]
        technologies_driver[iTb] = tech_activity in activities_driver
        technologies_energy[iTb] = tech_activity in activities_energy
        technologies_emission[iTb] = tech_activity in activities_emission

    # === Obtain the day ranges of flexible technologies ===
    flexibility_range_days = np.zeros(nTb)
    flexibility_mapping = {
        '1 day [d]': 1, '3 days [r]': 3, '1 week [w]': 7, '1 month [m]': 31,
        '1 season [s]': 92, '6 months [b]': 183, '1 year [y]': 365
    }
    for iTb, range_key in enumerate(flexibility_range):
        flexibility_range_days[iTb] = flexibility_mapping.get(range_key, 0)

    # === Calculate annuity factors ===
    print('--Calculating annuity factors')
    # Technologies that came from IESA-Opt (via a merged/unified database -
    # see mod0_load_duckdb._load_technologies) have no agent_profile on their
    # activity, since agent-based discounting is an IESA-Sim-only concept -
    # IESA-Opt carries its own per-technology wacc instead, so fall back to
    # that. A handful have neither (no agent, no wacc - e.g. Passenger Cars,
    # CO2 Air n-ETS) - for those, fall back to the median wacc of technologies
    # in the same category, since no per-technology financial data exists at
    # all for them.
    category_wacc = {}
    for cat in set(tech_categories):
        cat_wacc = tech_wacc[np.array(tech_categories) == cat]
        cat_wacc = cat_wacc[~np.isnan(cat_wacc)]
        if cat_wacc.size:
            category_wacc[cat] = np.median(cat_wacc)

    annuity_fact = np.zeros(nTb)
    annuity_fact_infra = np.zeros(nTi)
    for iTb in range(nTb):
        n = ec_lifetime[iTb]
        tech_activity = activity_per_tech[iTb]
        agent_tech = activity_agent[activity_names.index(tech_activity)]
        if agent_tech is not None:
            r = agents_dr[agent_profiles.index(agent_tech)]
        elif not np.isnan(tech_wacc[iTb]):
            r = tech_wacc[iTb]
        elif tech_categories[iTb] in category_wacc:
            r = category_wacc[tech_categories[iTb]]
        else:
            raise ValueError(
                f"Technology {tech_balancers[iTb]!r} has no agent_profile, no wacc, "
                f"and no other {tech_categories[iTb]!r}-category technology with a wacc "
                "to fall back to - cannot determine a discount rate for its annuity factor."
            )
        annuity_fact[iTb] = r / (1 - (1 + r) ** -n)
    for iTi in range(nTi):
        n = ec_lifetime_infra[iTi]
        coord_agent = agent_profiles.index('Medium companies')
        r = agents_dr[coord_agent]
        annuity_fact_infra[iTi] = r / (1 - (1 + r) ** -n) # MODIFIED & CHECK: The code used to say annuity_fact here, while it should be annuity_fact_infra - right?
        # annuity_fact[iTi] = r / (1 - (1 + r) ** -n)

    # === Initialize energy prices ===
    print('--Initializing energy prices')
    nAe = len(activities_energy)
    initialized_energy_prices = np.zeros(nAe)
    for iAe, energy_activity in enumerate(activities_energy):
        label = activity_label[activity_names.index(energy_activity)]
        coord_label = energy_labels.index(label)
        initialized_energy_prices[iAe] = energyPrice_init[coord_label]

    # === Initialize emission prices ===
    print('--Initializing emission prices')
    nAc = len(activities_emission)
    initialized_emission_prices = np.zeros(nAc)
    for iAc, emission_activity in enumerate(activities_emission):
        # An 'Emission'-type activity with no technology backing it at all
        # (activity_per_tech never contains it) is a real, if unusual,
        # dataset shape - a merged/unified database can declare more
        # emission activities than it has abatement technologies for, and
        # the investment logic elsewhere already tolerates this same gap
        # (prints "There is no main technology for activity: ..." and moves
        # on) rather than treating it as fatal. Leave this one at its
        # zero-initialized default instead of crashing on .index().
        if emission_activity not in activity_per_tech:
            print(f"--**No technology is associated with the emission activity: {emission_activity}")
            continue
        coord_tech = activity_per_tech.index(emission_activity)
        initialized_emission_prices[iAc] = np.max(vom_cost_init[coord_tech])

    # === Extracting infrastructure activities ===
    print('--Extracting infrastructure activities')
    coord_infra = np.zeros(len(activity_names), dtype=bool)
    for tech_activity in activity_per_tech_infra:
        coord_infra |= np.array(activity_names) == tech_activity
    activities_infra = np.array(activity_names)[coord_infra]

    # === Initialiying activity prices ===
    initialized_prices = initialize_prices(activities, tech_categories, activity_label,
                                       activity_per_tech, activities_energy, activities_emission,
                                       taxes_activities, taxes_values, vom_cost)

    # === Allocate the planned decommissioning === 
    print('--Allocating planned decommissioning')
    decommissionings = tech_stock_dec
    # This line of code shows remaining stock of technologies after planned decommissioning, excluding negative numbers (i.e. the lowest possible number is 0)
    decom_dif = np.maximum(0, tech_stock_exist - np.sum(decommissionings, axis=1)) # CHECK: I feel like there's a mistake in the code here? Why is decom_diff not used? In the matlab version, there's a comment: "% decommissionings(:,end) = decommissionings(:,end) + decom_dif;"

    # === Defining categories ===
    # Cost categories definition
    cost_categories = ['capital', 'fixed operational', 'variable', 'fuels', 'emissions', 'exporting revenues']

    # LCOPs categories definition
    categories_LCOPs = ['Investment', 'FOM', 'VOM', 'Fuels', 'Emissions']

    # Activities accounted for the emission targets
    emission_targets = ['CO2 Air ETS', 'CO2 Air n-ETS']

    # Policy cashflows categories
    policy_cashflows_categories = ['EUA', 'Taxes', 'Feed-In subsidies', 'Investment subsidies']

    # === Price percentile ranges and hours ===
    print('--Calculating price ranges and hours')
    price_ranges = np.linspace(0, 1, nRp)
    price_ranges_hours = np.zeros(nRp, dtype=int)
    denom = nRp - 1                              # 20
    nums  = np.arange(nRp, dtype=int) * nH       # [0, 8760, 17520, …, 175200]
    price_ranges_hours = (nums + denom - 1) // denom
    price_ranges_hours[0] = 1

    # === Initialize model dimensions === 
    print('--Initializing model dimensions')
    nIp = settings['iterations']['power']
    # The 3 lines below set nIp to an uneven number and ensure it's between 3 and 21
    if nIp % 2 == 0:
        nIp += 1
    nIp = min(max(nIp, nIp_min), nIp_max)

    dimensions = Struct(
        nH=nH,
        nDy=nDy,
        nHd=nHd,
        nIp=nIp,
        nIb=settings['iterations']['balancing'],
        nId=settings['iterations']['dispatch'],
        nP=len(periods),
        nTb=len(tech_balancers),
        nTd=np.sum(technologies_driver),
        nTc=np.sum(technologies_emission),
        nTi=len(tech_infra),
        nA=len(activity_names),
        nAe=len(activities_energy),
        nAd=len(activities_driver),
        nAc=len(activities_emission),
        nAk=len(activities_elec),
        nAg=len(activities_gaseous),
        nAi=len(activities_infra),
        nEl=len(energy_labels),
        nAT=len(agent_types),
        nMC=len(multi_criteria_categories),
        nCc=len(cost_categories),
        nTL=len(categories_LCOPs),
        nRp=nRp,
        nPc=len(policy_cashflows_categories),
    )

    # === Initialize model variables ===
    print('--Initializing model variables')
    energy_prices = np.zeros((dimensions['nAe'], dimensions['nP']))
    energy_scarcity = np.zeros((dimensions['nAe'], dimensions['nP']))
    emission_prices = np.zeros((dimensions['nAc'], dimensions['nP']))
    prices_hourly = np.zeros((dimensions['nH'], dimensions['nA'], dimensions['nP']))
    energy_prices_hourly = np.zeros((dimensions['nH'], dimensions['nAe'], dimensions['nP']))
    energy_prices_ranges = np.zeros((dimensions['nRp'], dimensions['nAe'], dimensions['nP']))

    tech_stock = np.zeros((dimensions['nTb'], dimensions['nP']))
    tech_use = np.zeros((dimensions['nTb'], dimensions['nP']))
    tech_use_hourly = np.zeros((dimensions['nH'], dimensions['nTb'], dimensions['nP']))
    tech_stock_infra = np.zeros((dimensions['nTi'], dimensions['nP']))
    investments_infra = np.zeros((dimensions['nTi'], dimensions['nP']))
    investments = np.zeros((dimensions['nTb'], dimensions['nP']))
    retrofittings = np.zeros((dimensions['nTb'], dimensions['nP']))

    tech_LCOPs = np.zeros((dimensions['nTb'], dimensions['nP']))
    tech_LCOPs_matrix = np.zeros((dimensions['nTb'], dimensions['nTL'], dimensions['nP']))

    multi_criteria_performance_tech = np.zeros((dimensions['nTb'], dimensions['nMC'], dimensions['nP']))
    tech_choices_agent = np.zeros((dimensions['nTb'], dimensions['nAT'], dimensions['nP']))
    tech_choices_LCOP_order = np.zeros((dimensions['nTb'], dimensions['nP']), dtype=int)

    primary_energy = np.zeros((dimensions['nEl'], dimensions['nP']))
    energy_exports = np.zeros((dimensions['nAe'], dimensions['nP']))

    emissions = np.zeros((dimensions['nP'],1))
    emissions_sector_pos = np.zeros((len(sectors), dimensions['nP']))
    emissions_sector_neg = np.zeros((len(sectors), dimensions['nP']))
    emissions_stored = np.zeros((dimensions['nP'],1))

    system_costs = np.zeros((dimensions['nCc'], dimensions['nP']))

    generator_norm_ut_fact = np.zeros((dimensions['nTb'], dimensions['nP']))
    generator_capt_rate = np.zeros((dimensions['nTb'], dimensions['nP']))
    generator_cash_flow = np.zeros((dimensions['nTb'], dimensions['nP']))

    policy_cashflows = np.zeros((dimensions['nPc'], dimensions['nP']))

    # === Saving outputs ===
    types.energy.primary = primary_energy
    types.policy_cashflows_categories = policy_cashflows_categories

    activities.drivers.names = activities_driver
    activities.energies.names = activities_energy
    activities.emissions.names = activities_emission
    activities.emissions.targets = emission_targets
    activities.electricity.names = activities_elec
    activities.electricity.coords = activities_elec_coord
    activities.gaseous.names = activities_gaseous
    activities.gaseous.coords = activities_gaseous_coord
    activities.infra.names = activities_infra
    activities.energies.prices = Struct(
        yearly=energy_prices,
        initialized=initialized_energy_prices,
        hourly=energy_prices_hourly,
        ranges=energy_prices_ranges,
        price_ranges=price_ranges,
        price_ranges_hours=price_ranges_hours,
    )
    activities.energies.scarcity = energy_scarcity
    activities.emissions.prices = Struct(
        yearly=emission_prices,
        initialized=initialized_emission_prices,
    )
    activities.prices = Struct(
        initialized=initialized_prices,
        hourly=prices_hourly,
    )

    technologies.balancers.drivers.names = technologies_driver
    technologies.balancers.energies.names = technologies_energy
    technologies.balancers.emissions.names = technologies_emission
    technologies.balancers.stocks.evolution = tech_stock
    technologies.balancers.use = Struct(
        yearly=tech_use,
        hourly=tech_use_hourly,
    )
    technologies.balancers.investments = investments
    technologies.balancers.retrofittings = retrofittings
    technologies.balancers.decommissionings = decommissionings
    technologies.balancers.generators = Struct(
        NUF=generator_norm_ut_fact,
        CR=generator_capt_rate,
        CF=generator_cash_flow,
    )
    technologies.balancers.flexibility.range_days = flexibility_range_days
    technologies.balancers.costs.annuity = annuity_fact
    technologies.balancers.lcops = Struct(
        categories=categories_LCOPs,
        values=tech_LCOPs,
        matrix=tech_LCOPs_matrix,
    )
    technologies.balancers.mca.matrix = multi_criteria_performance_tech
    technologies.balancers.choices_agent = tech_choices_agent
    technologies.balancers.choices_lcop_order = tech_choices_LCOP_order
    technologies.infra.stocks.evolution = tech_stock_infra
    technologies.infra.investments = investments_infra
    technologies.infra.costs.annuity = annuity_fact_infra

    results = Struct(
        emissions=emissions,
        emissions_sector_pos=emissions_sector_pos,
        emissions_sector_neg=emissions_sector_neg,
        emissions_stored=emissions_stored,
        primary=primary_energy,
        exports=energy_exports,
        costs=Struct(
            categories=cost_categories,
            system=system_costs,
        ),
        policy_cashflows=policy_cashflows,
    )

    return dimensions, types, activities, technologies, results
