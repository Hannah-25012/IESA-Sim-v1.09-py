import numpy as np

# Attention: Still have to check the values properly, 2 values for generator_CaptRate don't match

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
    # debugging
    # print(techUse_hourly)
    profileType = profiles['types']
    hourly_profiles = profiles['shapes']

    # Preallocate arrays for normalized utilization factors, capture rates, and cash flows
    generator_NormUtFact = np.zeros(nTb)
    generator_CaptRate = np.zeros(nTb)
    generator_CashFlow = np.zeros(nTb)

    # Loop over each technology
    for iTb in range(nTb):
        # Identify characteristics of the technology using list comprehensions
        activity_indices = np.where(np.array([act == activityPer_tech[iTb] for act in activities_names]))[0]
        if activity_indices.size == 0:
            raise ValueError(f"No matching activity found for technology index {iTb}")
        if activity_indices.size > 1:
            print(f"Warning: Multiple matches found for technology index {iTb}. Using the first match.")
        activity_index = activity_indices[0]

        profile_indices = np.where(np.array([pt == hourly_profile_tech[iTb] for pt in profileType]))[0] 
        # Flatten the array to mimic MATLAB’s column-major (Fortran-style) linear indexing. Note: NumPy uses row-major order by default, so you need to use 'F' order if you want exact MATLAB behavior.
        flat_hourly_profiles = hourly_profiles.flatten(order='F')
        selected_value = flat_hourly_profiles[profile_indices[0]]

        # Compute normalized utility factor
        denom = techStock[iTb] * cap2act[iTb] * selected_value
        # `frac` may contain np.inf or np.nan if denom was zero.
        with np.errstate(divide='ignore', invalid='ignore'):
            frac = techUse_hourly[:, iTb] / denom
        generator_NormUtFact[iTb] = np.mean(frac)

        # Compute the cash flow
        cash_flow_vector = prices_hourly @ activity_balances[iTb, :]
        generator_CashFlow[iTb] = np.sum(techUse_hourly[:, iTb] * cash_flow_vector)

        # Compute generator revenues for the selected activity
        generator_revenues = np.sum(prices_hourly[:, activity_index] * techUse_hourly[:, iTb])

        # Compute the capture rate
        # Note: tech_use_sum is the problem - here the values for matlab 467, 468 don't match
        tech_use_sum = np.sum(techUse_hourly[:, iTb])
        # debugging
        # print ("tech_use_sum:", tech_use_sum, "for:", iTb)
        prices_mean = np.mean(prices_hourly[:, activity_index])
        # debugging
        # print ("prices_mean:", prices_mean, "for:", iTb)
        if tech_use_sum == 0 or prices_mean == 0:
            generator_CaptRate[iTb] = np.nan
        else:
            generator_CaptRate[iTb] = generator_revenues / tech_use_sum / prices_mean

    # Save computed values back into the technologies structure
    # values seem ok, still have to check the sum (need to figure out how to print with NaN values).
    technologies['balancers']['generators']['NUF'][:, iP] = generator_NormUtFact
    # Attention: matlab 467 and 468 are different. Still have to fix this. All the other values seem to match.
    technologies['balancers']['generators']['CR'][:, iP] = generator_CaptRate
    # values seem ok, still have to check sum (need to figure out how to print with Nan values)
    technologies['balancers']['generators']['CF'][:, iP] = generator_CashFlow

    # debugging
    # print("sum of generator_CaptRate:", np.sum(generator_CaptRate))
    # print("sum of generator_CashFlow:", np.sum(generator_CashFlow))

    return technologies
