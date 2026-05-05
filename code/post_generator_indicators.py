# File to obtain the generators' key indicators
import numpy as np

def post_generator_indicators(dimensions, activities, technologies, profiles, iP):

    # Extract parameters
    nTb = dimensions['nTb']
    activities_names = activities['names']
    prices_hourly = activities['prices']['hourly'][:, :, iP]
    activityPer_tech = technologies['balancers']['activities']
    activity_balances = technologies['balancers']['activity_balances']
    techStock = technologies['balancers']['stocks']['evolution'][:, iP]
    cap2act = technologies['balancers']['cap2acts']
    hourly_profile_tech = technologies['balancers']['profiles']
    techUse_hourly = technologies['balancers']['use']['hourly'][:, :, iP]
    profileType = profiles['types']
    hourly_profiles = profiles['shapes']

    # Obtain the normalized utilization factors using price and availability shapes
    generator_NormUtFact = np.zeros(nTb) # Preallocate
    generator_CaptRate = np.zeros(nTb)
    generator_CashFlow = np.zeros(nTb)

    # Loop to obtain indicators per technology
    for iTb in range(nTb):

        # Identify characteristics of the technology
        activity_indices = np.where(np.array([act == activityPer_tech[iTb] for act in activities_names]))[0]
        if activity_indices.size == 0:
            raise ValueError(f"No matching activity found for technology index {iTb}")
        if activity_indices.size > 1:
            print(f"Warning: Multiple matches found for technology index {iTb}. Using the first match.")
        activity_index = activity_indices[0]
        profile_indices = np.where(np.array([pt == hourly_profile_tech[iTb] for pt in profileType]))[0] 
        flat_hourly_profiles = hourly_profiles.flatten(order='F') # Flatten the array to mimic MATLAB’s column-major (Fortran-style) linear indexing. Note: NumPy uses row-major order by default, so you need to use 'F' order if you want exact MATLAB behavior.
        selected_value = flat_hourly_profiles[profile_indices[0]]

        # Obtain the indicators
        # Compute the normalized utilization factor
        denom = techStock[iTb] * cap2act[iTb] * selected_value
        with np.errstate(divide='ignore', invalid='ignore'): # `frac` may contain np.inf or np.nan if denom was zero.
            frac = techUse_hourly[:, iTb] / denom
        generator_NormUtFact[iTb] = np.mean(frac)

        # Compute the cash flow
        cash_flow_vector = prices_hourly @ activity_balances[iTb, :]
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
    technologies['balancers']['generators']['NUF'][:, iP] = generator_NormUtFact # CHECK: incorrect bec of sum of tech_use_hourly
    technologies['balancers']['generators']['CR'][:, iP] = generator_CaptRate
    technologies['balancers']['generators']['CF'][:, iP] = generator_CashFlow # CHECK: incorrect bec of sum of tech_use_hourly

    return technologies
