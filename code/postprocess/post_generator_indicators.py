# File to obtain the generators' key indicators
import numpy as np

def post_generator_indicators(dimensions, activities, technologies, profiles, iP):

    # Extract parameters
    nTb = dimensions['nTb']
    tech_entities = technologies.balancers.entities
    prices_hourly = activities.prices.hourly[:, :, iP]
    techStock = technologies.balancers.stocks.evolution[:, iP]
    cap2act = technologies.balancers.cap2acts
    techUse_hourly = technologies.balancers.use.hourly[:, :, iP]
    profileType = profiles.types
    hourly_profiles = profiles.shapes

    # Profile-type -> position lookup, resolved once instead of re-scanning
    # profileType for every technology.
    profile_idx_by_type = {name: i for i, name in enumerate(profileType)}
    flat_hourly_profiles = hourly_profiles.flatten(order='F') # Flatten the array to mimic MATLAB's column-major (Fortran-style) linear indexing. Note: NumPy uses row-major order by default, so you need to use 'F' order if you want exact MATLAB behavior.
    # Emission-type activities are now stored positive-for-emitters (IESA-Opt
    # convention) - used below to un-negate just those entries of the balance
    # vector fed into cash_flow_vector, which feeds invest_power_technologies.py's
    # payback-time investment decision and must stay numerically identical.
    coord_emission_act = np.array([a.is_emission for a in activities.entities])

    # Obtain the normalized utilization factors using price and availability shapes
    generator_NormUtFact = np.zeros(nTb) # Preallocate
    generator_CaptRate = np.zeros(nTb)
    generator_CashFlow = np.zeros(nTb)

    # Loop to obtain indicators per technology
    for tech in tech_entities:
        iTb = tech.idx

        # Identify characteristics of the technology
        if tech.activity is None:
            raise ValueError(f"No matching activity found for technology index {iTb}")
        activity_index = tech.activity.idx
        profile_idx = profile_idx_by_type[tech.profile]
        selected_value = flat_hourly_profiles[profile_idx]

        # Obtain the indicators
        # Compute the normalized utilization factor
        denom = techStock[iTb] * cap2act[iTb] * selected_value
        with np.errstate(divide='ignore', invalid='ignore'): # `frac` may contain np.inf or np.nan if denom was zero.
            frac = techUse_hourly[:, iTb] / denom
        generator_NormUtFact[iTb] = np.mean(frac)

        # Compute the cash flow. tech.activity_balances is a VIEW into the
        # shared activity_balances matrix (see entities.py) - copy before
        # negating so this doesn't mutate the real array.
        balance_for_cashflow = tech.activity_balances.copy()
        balance_for_cashflow[coord_emission_act] = -balance_for_cashflow[coord_emission_act]
        cash_flow_vector = prices_hourly @ balance_for_cashflow
        generator_CashFlow[iTb] = np.sum(techUse_hourly[:, iTb] * cash_flow_vector)

        # Compute generator revenues for the selected activity
        generator_revenues = np.sum(prices_hourly[:, activity_index] * techUse_hourly[:, iTb])

        # Compute the capture rate
        tech_use_sum = np.sum(techUse_hourly[:, iTb])
        prices_mean = np.mean(prices_hourly[:, activity_index])
        if tech_use_sum == 0 or prices_mean == 0:
            generator_CaptRate[iTb] = np.nan
        else:
            generator_CaptRate[iTb] = generator_revenues / tech_use_sum / prices_mean

    # Save computed values back into the technologies structure
    technologies.balancers.generators.NUF[:, iP] = generator_NormUtFact # CHECK: incorrect bec of sum of tech_use_hourly
    technologies.balancers.generators.CR[:, iP] = generator_CaptRate
    technologies.balancers.generators.CF[:, iP] = generator_CashFlow # CHECK: incorrect bec of sum of tech_use_hourly

    return technologies
