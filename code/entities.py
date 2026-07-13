# Object-relational view over the simulation data: Activity/Technology
# instances with real cross-references (technology.activity, activity.technologies),
# replacing the pattern of parallel arrays plus hand-rolled index-hunting
# (`iAc = [i for i, t in enumerate(activity_types) if t == 'Emission']`,
# `activities_names.index(activity_per_tech[iTb])`, etc.) that's spread across
# the invest_*/disp_*/post_* files.
#
# Per-period/per-hour numeric data (costs, stocks, activity balances, ...) stays
# backed by the shared numpy matrices in the Struct model
# (technologies.balancers.costs.investments etc.): each entity's numeric fields
# are *views* into those matrices (a fixed-row/column numpy slice is a view, not
# a copy), so code still using the raw matrices and code using entities see and
# mutate the same underlying data. This keeps the hourly/period dispatch loops
# vectorizable while giving the invest/post logic a relational way to navigate
# the data instead of re-deriving index lists everywhere.


class Activity:
    def __init__(self, idx, name, resolution, type_, label, agent_profile_name, volumes):
        self.idx = idx
        self.name = name
        self.resolution = resolution
        self.type = type_
        self.label = label
        self.agent_profile_name = agent_profile_name
        self.volumes = volumes  # view: 1D over periods

        # Filled in by link_entities()
        self.technologies = []           # Technology objects with .activity is this
        self.infrastructure = []         # Infrastructure objects with .activity is this
        self.flexible_technologies = []  # Technology objects benefiting from flexibility here
        self.energy_idx = None           # position within activities.energies.names, or None
        self.elec_idx = None             # position within activities.electricity.names, or None

    @property
    def is_driver(self):
        return self.type == 'Driver'

    @property
    def is_energy(self):
        return self.type in ('Energy', 'Fix Energy')

    @property
    def is_emission(self):
        return self.type == 'Emission'

    @property
    def is_electricity(self):
        return self.label == 'Electricity'

    @property
    def is_gaseous(self):
        return self.resolution == 'daily'

    def __repr__(self):
        return f"Activity({self.name!r})"


class Technology:
    def __init__(self, idx, id_, category, sector, subsector, unit, name, activity_name,
                 cap2act, lifetime, dispatch, profile,
                 social_perception, complexity, subsidy_subject, feedin_subject,
                 shedding_capacity, shedding_limits, shedding_guarantee,
                 flexibility_form, flexibility_activity_name, flexibility_capacity,
                 flexibility_volume, flexibility_range, flexibility_losses, flexibility_nonnegotiable,
                 buffer_up, buffer_down, buffer_capacity,
                 stock_deploy, stock_initial, stock_dec_planned, stock_min, stock_max,
                 costs_investment, costs_fom, costs_vom,
                 activity_balances_row):
        self.idx = idx
        self.id = id_
        self.category = category
        self.sector = sector
        self.subsector = subsector
        self.unit = unit
        self.name = name

        self.activity_name = activity_name
        self.activity = None  # Activity, resolved by link_entities()

        self.cap2act = cap2act
        self.lifetime = lifetime
        self.dispatch = dispatch
        self.profile = profile

        self.social_perception = social_perception
        self.complexity = complexity
        self.subsidy_subject = subsidy_subject
        self.feedin_subject = feedin_subject

        self.shedding_capacity = shedding_capacity
        self.shedding_limits = shedding_limits
        self.shedding_guarantee = shedding_guarantee

        self.flexibility_form = flexibility_form
        self.flexibility_activity_name = flexibility_activity_name
        self.flexibility_activity = None  # Activity, resolved by link_entities() if present
        self.flexibility_capacity = flexibility_capacity
        self.flexibility_volume = flexibility_volume
        self.flexibility_range = flexibility_range
        self.flexibility_losses = flexibility_losses
        self.flexibility_nonnegotiable = flexibility_nonnegotiable

        self.buffer_up = buffer_up
        self.buffer_down = buffer_down
        self.buffer_capacity = buffer_capacity

        self.stock_deploy = stock_deploy
        self.stock_initial = stock_initial
        self.stock_dec_planned = stock_dec_planned  # view: 1D over periods
        self.stock_min = stock_min                  # view: 1D over periods
        self.stock_max = stock_max                  # view: 1D over periods

        self.costs_investment = costs_investment  # view: 1D over periods
        self.costs_fom = costs_fom                # view: 1D over periods
        self.costs_vom = costs_vom                # view: 1D over periods

        self.activity_balances = activity_balances_row  # view: 1D over ALL activities
        self.retrofit_options = []  # Technology objects this can retrofit into

    @property
    def is_buffer(self):
        return self.buffer_capacity != 0

    def balance_with(self, activity):
        """This technology's coupling coefficient for a given Activity."""
        return self.activity_balances[activity.idx]

    def __repr__(self):
        return f"Technology({self.id!r})"


class Infrastructure:
    def __init__(self, idx, id_, category, name, unit, activity_name,
                 cap2act, lifetime, stock_initial, costs_investment, costs_fom):
        self.idx = idx
        self.id = id_
        self.category = category
        self.name = name
        self.unit = unit

        self.activity_name = activity_name
        self.activity = None  # Activity, resolved by link_entities()

        self.cap2act = cap2act
        self.lifetime = lifetime
        self.stock_initial = stock_initial
        self.costs_investment = costs_investment  # view: 1D over periods
        self.costs_fom = costs_fom                # view: 1D over periods

    def __repr__(self):
        return f"Infrastructure({self.id!r})"


def link_entities(activities, technologies):
    """Build the Activity/Technology/Infrastructure object graph from the
    Struct-based data (post mod1_initialize), resolving foreign keys into
    real object references.

    Returns (activity_list, tech_list, infra_list), each ordered to match the
    underlying arrays (activity_list[i].idx == i, matching activities.names[i],
    etc.).
    """
    activity_list = []
    for i, name in enumerate(activities.names):
        activity_list.append(Activity(
            idx=i, name=name,
            resolution=activities.resolution[i],
            type_=activities.types[i],
            label=activities.labels[i],
            agent_profile_name=activities.agents[i],
            volumes=activities.volumes[i, :],
        ))
    by_name = {a.name: a for a in activity_list}

    energy_by_name = {name: i for i, name in enumerate(activities.energies.names)}
    elec_by_name = {name: i for i, name in enumerate(activities.electricity.names)}
    for a in activity_list:
        a.energy_idx = energy_by_name.get(a.name)
        a.elec_idx = elec_by_name.get(a.name)

    balancers = technologies.balancers
    tech_list = []
    for i, tech_id in enumerate(balancers.ids):
        tech_list.append(Technology(
            idx=i, id_=tech_id,
            category=balancers.categories[i], sector=balancers.sectors[i],
            subsector=balancers.subsectors[i], unit=balancers.units[i],
            name=balancers.names[i], activity_name=balancers.activities[i],
            cap2act=balancers.cap2acts[i], lifetime=balancers.costs.lifetimes[i],
            dispatch=balancers.dispatch[i], profile=balancers.profiles[i],
            social_perception=balancers.agents.social_perception[i],
            complexity=balancers.agents.complexity[i],
            subsidy_subject=balancers.policies.subsidy_subject[i],
            feedin_subject=balancers.policies.feedin_subject[i],
            shedding_capacity=balancers.shedding.capacity[i],
            shedding_limits=balancers.shedding.limits[i],
            shedding_guarantee=balancers.shedding.guarantee[i],
            flexibility_form=balancers.flexibility.form[i],
            flexibility_activity_name=balancers.flexibility.activity[i],
            flexibility_capacity=balancers.flexibility.capacity[i],
            flexibility_volume=balancers.flexibility.volume[i],
            flexibility_range=balancers.flexibility.range[i],
            flexibility_losses=balancers.flexibility.losses[i],
            flexibility_nonnegotiable=balancers.flexibility.nonnegotiable[i],
            buffer_up=balancers.buffers.up[i], buffer_down=balancers.buffers.down[i],
            buffer_capacity=balancers.buffers.capacity[i],
            stock_deploy=balancers.stocks.deploy[i], stock_initial=balancers.stocks.initial[i],
            stock_dec_planned=balancers.stocks.dec_planned[i, :],
            stock_min=balancers.stocks.min[i, :], stock_max=balancers.stocks.max[i, :],
            costs_investment=balancers.costs.investments[i, :],
            costs_fom=balancers.costs.foms[i, :], costs_vom=balancers.costs.voms[i, :],
            activity_balances_row=balancers.activity_balances[i, :],
        ))
    by_id = {t.id: t for t in tech_list}

    for tech in tech_list:
        tech.activity = by_name.get(tech.activity_name)
        if tech.activity is not None:
            tech.activity.technologies.append(tech)

        if tech.flexibility_activity_name is not None:
            tech.flexibility_activity = by_name.get(tech.flexibility_activity_name)
            if tech.flexibility_activity is not None:
                tech.flexibility_activity.flexible_technologies.append(tech)

    infra = technologies.infra
    infra_list = []
    for i, infra_id in enumerate(infra.ids):
        infra_list.append(Infrastructure(
            idx=i, id_=infra_id,
            category=infra.categories[i], name=infra.names[i], unit=infra.units[i],
            activity_name=infra.activity[i],
            cap2act=infra.cap2acts[i], lifetime=infra.costs.lifetimes[i],
            stock_initial=infra.stocks.initial[i],
            costs_investment=infra.costs.investments[i, :], costs_fom=infra.costs.foms[i, :],
        ))
    for infra_obj in infra_list:
        infra_obj.activity = by_name.get(infra_obj.activity_name)
        if infra_obj.activity is not None:
            infra_obj.activity.infrastructure.append(infra_obj)

    for from_id, to_id in zip(technologies.retrofittings["from"], technologies.retrofittings["to"]):
        from_tech = by_id.get(from_id)
        to_tech = by_id.get(to_id)
        if from_tech is not None and to_tech is not None:
            from_tech.retrofit_options.append(to_tech)

    return activity_list, tech_list, infra_list
