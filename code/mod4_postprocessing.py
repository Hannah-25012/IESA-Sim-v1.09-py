# Function to obtain the post-processed indicators to report.

from post_generator_indicators import post_generator_indicators
from post_prices import post_prices
from post_primary_energy import post_primary_energy

def mod4_postprocessing(dimensions, parameters, types, activities, technologies, profiles, policies, results, iP):

    # Obtain generators indicators
    # Attention: Not all values match, still have to fix this
    print('--Calculating the generators indicators ...')
    technologies = post_generator_indicators(dimensions, activities, technologies, profiles, iP)

    # Obtaining energy and emission prices
    print('--Obtaining energy and emission prices...')
    activities = post_prices(dimensions, parameters, activities, technologies, policies, iP)

    # Obtaining primary energy
    print('--Calculating primary energy...')
    results = post_primary_energy(dimensions, types, activities, technologies, results, iP)

    return activities, technologies, results
