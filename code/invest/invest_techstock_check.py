# Function to check the sufficiency of the new technology stocks
# CHECK: It seems like this function is unfinished, since so many things aren't used. Not sure what's going on here?
import numpy as np
from invest_drivers_sufficiency import invest_drivers_sufficiency

def invest_techstock_check(dimensions, activities, technologies,
                          tech_stock_new, tech_stock_exist, report_stocks, iP):

    # Extract Parameters
    nTd = dimensions['nTd']
    tech_entities = technologies.balancers.entities
    tech_stock_max = technologies.balancers.stocks.max[:, iP]

    # Check that there are no negative tech stocks
    tech_stock_new = np.maximum(tech_stock_new, 0)

    # Check that the new stocks are not violating max stocks
    if report_stocks:
        print("---The determined driver investments are shown below:")
        print("    Tech ID     Old Stock    New Stock    Max Stock    Dif Stock")

    for tech in tech_entities[:nTd]:
        if tech_stock_max[tech.idx] < tech_stock_new[tech.idx]:
            tech_stock_new[tech.idx] = tech_stock_max[tech.idx]

        if report_stocks:
            print(f"{tech.id:>12} {tech_stock_exist[tech.idx]:>12.2f} \
                   {tech_stock_new[tech.idx]:>12.2f} {tech_stock_max[tech.idx]:>12.2f} \
                   {tech_stock_max[tech.idx] - tech_stock_new[tech.idx]:>12.2f}")

    # Check the gaps and report them
    invest_drivers_sufficiency(dimensions, activities, technologies,
                               tech_stock_exist, report_stocks, iP)

    return tech_stock_new
