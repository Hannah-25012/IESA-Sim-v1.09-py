class Parameters:
    powinv_SPBT_benchmark = "Power investments simple payback time benchmark"
    powinv_SPBT_min = "Power investments simple payback time minimum"
    powinv_CR_threshold = "Power investments capture rate threshold"
    powinv_CR_min = "Power investments capture rate minimum"
    powinv_NUF_threshold = "Power investments normalized utilization factor threshold"
    powinv_NUF_min = "Power investments normalized utilization factor minimum"
    scarcity_penalization = "Scarcity penalization parameter"
    gas_premium = "Gas price premium spread under high demand"
    voll_value = "Value of lost load"
    voll_factor = 3.6
    min_spread_value= "Minimum battery spread"
    min_spread_factor=3.6
    gov_dr = "Government depreciation rate"
    exports_value = "Exports value as ratio of crude oil (0 for no)"

class Activities:
    activities_names="Activities"
    periods_start="Evolution of volumes in time"
    activity_resolution="Dispatch resolution"
    activity_type_act="Activity Type"
    activity_label="Energy Label"
    activity_agent="Agent Profile"

class Agents:
    types = "Agent Types"
    profiles = "Agent profiles"
    ic_criteria =  "Weights for investment criteria"
    rates = "Expected rates of return"

class Types:
    activity_type = "Activity Types"
    sectors = "Sectors"
    energy_labels = "Energy Labels"
    energy_price_init = "Prices Initialization"

class Retrofitting:
    tech_id_original = "Tech_ID Original "
    tech_id_new = "Tech_ID New"
    enabled = "Enabled\n[y/n]"
    investment_cost = "Overnight retrofitting investment [M€/UoC]"

class HourlyProfiles:
    hour = "hour"
    day = "day"
    month = "month"

class PriceProfiles:
    interconnector = "Electricity IC"

class Technologies:
    # Column headers are read from a 2-row header (group / field), so most
    # names below are "<group> / <field>" strings as they appear once flattened.
    tech_id = "Tech. Specifics / Tech_ID"
    category = "Tech. Specifics / Category"
    sector = "Tech. Specifics / Sector"
    subsector = "Tech. Specifics / Sub-sector"
    main_activity = "Tech. Specifics / Main Activity"
    name = "Tech. Specifics / Name"
    unit = "Tech. Specifics / UoC"
    investment = "Cost data / Investment "
    fixed_om = "Cost data / Fixed O&M "
    variable_om = "Cost data / Variable O&M "
    ec_lifetime = "Cost data / Ec. Lifetime"
    cap2act = "Operation data / Cap2Act"
    dispatch_type = "Operation data / Type of process"
    hourly_profile = "Operation data / Type of profile"
    social_perception = "Agents parameters / Social perception"
    perceived_complexity = "Agents parameters / Perceived complexity"
    subsidy_subject = "Subsidies influence / Subject to investment subsidy"
    feedin_subject = "Subsidies influence / Subject to feed-in tariff subsidy"
    shedding_capacity = "Asymetric flexibility / Shedding capacity"
    shedding_volume = "Asymetric flexibility / Shedding volume"
    shedding_guarantee = "Asymetric flexibility / Contract guaranteed volume"
    flexibility_form = "Flexibility data / Form of Flexibility"
    flexibility_activity = "Flexibility data / Benefited activity"
    flexibility_capacity = "Flexibility data / Flexible installed capacity"
    flexibility_volume = "Flexibility data / Storage capacity / (hours of charge)"
    flexibility_range = "Flexibility data / Shifting range"
    flexibility_losses = "Flexibility data / Losses"
    flexibility_nonnegotiable = "Flexibility data / Non-negotiable load"
    buffer_up = "Network buffers / Upward capacity"
    buffer_down = "Network buffers / Downward capacity"
    buffer_capacity = "Network buffers / Buffer capacity / (in relation to discharge)"
    tech_stock_deploy = "Technology Potentials / Maximum technology deployment"
    tech_stock_exist = "Technology Potentials / Current Installed Capacity"
    # The planned-decommissioning / min-stock / max-stock blocks that follow
    # tech_stock_exist in the sheet have no distinct column name of their own
    # (only ordinal markers or data-like fraction pairs) - their positions are
    # derived from tech_stock_exist's column position instead of a hardcoded
    # Excel letter range.

class Infrastructure:
    tech_id = "Tech. Specifics / Tech_ID"
    category = "Tech. Specifics / Category"
    name = "Tech. Specifics / Name"
    unit = "Tech. Specifics / UoC"
    investment = "Cost data / Investment "
    fixed_om = "Cost data / Fixed O&M "
    ec_lifetime = "Cost data / Ec. Lifetime"
    cap2act = "Other data / Cap2Act"
    activity = "Other data / Activity Constrained"
    tech_stock_exist = "Existing / 2020"
    planned_decommissioning = "Planned decommisioning"
    stock_min = "Minimum stock in a year"
    stock_max = "Maximum stock in a year"








