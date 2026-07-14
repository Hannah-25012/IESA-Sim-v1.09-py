# File to determine the dispatch of energy on an intrayearly basis
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
    
    return tech_use_hourly, prices_hourly, tech_stock
