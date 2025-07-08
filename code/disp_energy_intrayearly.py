import time
import numpy as np

from disp_power import disp_power
from disp_gas import disp_gas

def disp_energy_intrayearly(dimensions, parameters, activities, technologies, 
                             profiles, policies, tech_use_hourly, prices_hourly, iP, nId):

    # Solve the power system dispatch
    if nId:
        print('---Solving the power system dispatch...')

    # Time counter
    t_opt = time.time()
    tech_use_hourly, prices_hourly = disp_power(dimensions, parameters, activities, technologies, 
                                                profiles, tech_use_hourly, prices_hourly, iP, nId)
    # debugging (Note: values here in iId = 1 for technologies 466 and 467 are still correct. Only minor difference in last decimals of the overall sum.)
    # tech_use_hourly_sum = np.sum(tech_use_hourly, axis=0)
    # print("sum of sum of tech_use_hourly:", np.sum(tech_use_hourly_sum))

    # Time counter report
    if nId:
        elapsed_time = time.time() - t_opt
        print(f'---The elapsed time for the power system dispatch was: {elapsed_time:6.2f} seconds.')

    # Solve the dispatch of gaseous networks
    if nId:
        print('---Solving the gaseous networks dispatch...')

    # Time counter
    t_opt = time.time()
    tech_use_hourly, prices_hourly, tech_stock = disp_gas(dimensions, parameters, activities, technologies, 
                                                          profiles, policies, tech_use_hourly, prices_hourly, iP)
    # debugging (Note: this is where values for tech_use_hourly for 466, 467 in iId = 1 don't match. This is because the values for hourly_balance in disp_gas are off.)
    # tech_use_hourly_sum = np.sum(tech_use_hourly, axis=0)
    # print("sum of tech_use_hourly per technology:", tech_use_hourly_sum)
    # print("sum of sum of tech_use_hourly:", np.sum(tech_use_hourly_sum))

    # Time counter report
    if nId:
        elapsed_time = time.time() - t_opt
        print(f'---The elapsed time for the gaseous networks dispatch was: {elapsed_time:6.2f} seconds.')

    # Do some sanity checks on quality of the content
    # Check if there are NaNs in the tech_use_hourly matrix
    if np.isnan(tech_use_hourly).any():
        print('!!!NaN values exist within tech_use_hourly!!!')
        time.sleep(5)

    # Check if there are NaNs in the prices_hourly
    if np.isnan(prices_hourly).any():
        print('!!!NaN values exist within prices_hourly!!!')
        time.sleep(5)

    # debugging 
    # tech_stock_sum = np.sum(tech_stock)
    # Sum along the rows (axis=0)
    # tech_use_sum = np.sum(tech_use_hourly, axis=0)
    # prices_sum = np.sum(prices_hourly, axis=0)

    # Print the results
    # print ("Sum of techStock:", tech_stock_sum)
    # print("Sum of tech_use_hourly across rows:", tech_use_sum)
    # print("Sum of prices_hourly across rows:", prices_sum)
    
    return tech_use_hourly, prices_hourly, tech_stock
