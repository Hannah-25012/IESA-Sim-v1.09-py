# Function to check the sufficiency of the new technological stocks
# Manuel: It seems like this function is unfinished, since so many things aren't used. Not sure what's going on here?

import numpy as np
from invest_drivers_sufficiency import invest_drivers_sufficiency

def invest_techstock_check(dimensions, activities, technologies, 
                          tech_stock_new, tech_stock_exist, report_stocks, iP):
    
    # Extract Parameters
    nTd = dimensions['nTd']
    activities_names = activities['names']
    activities_driver = activities['drivers']['names']
    activities_net_volumes = activities['volumes'][:, iP]
    tech_balancers = technologies['balancers']['ids']
    activity_per_tech = technologies['balancers']['activities']
    cap2act = technologies['balancers']['cap2acts']
    tech_stock_max = technologies['balancers']['stocks']['max'][:, iP]

    # Check that there are no negative tech stocks
    # tech_stock_new = [max(stock, 0) for stock in tech_stock_new]
    tech_stock_new = np.maximum(tech_stock_new, 0)

    # Check that the new stocks are not violating max stocks
    if report_stocks:
        print("---The determined driver investments are shown below:")
        print("    Tech ID     Old Stock    New Stock    Max Stock    Dif Stock")

    for iTb in range(nTd):
        if tech_stock_max[iTb] < tech_stock_new[iTb]:
            tech_stock_new[iTb] = tech_stock_max[iTb]

        if report_stocks:
            print(f"{tech_balancers[iTb]:>12} {tech_stock_exist[iTb]:>12.2f} \
                   {tech_stock_new[iTb]:>12.2f} {tech_stock_max[iTb]:>12.2f} \
                   {tech_stock_max[iTb] - tech_stock_new[iTb]:>12.2f}")

    # Check the gaps and report them
    invest_drivers_sufficiency(dimensions, activities, technologies, 
                               tech_stock_exist, report_stocks, iP)

    return tech_stock_new
