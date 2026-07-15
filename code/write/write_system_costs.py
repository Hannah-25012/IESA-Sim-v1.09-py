# File to write system costs
# CHECK: Compare output from matlab & python
import pandas as pd

def write_system_costs(activities, results, writer):

    # Export Parameters
    periods = activities.periods
    cost_categories = results.costs.categories
    system_costs = results.costs.system

    # Sheet name
    sheet_name = 'System_costs'

    df = pd.DataFrame(system_costs / 1000, index=cost_categories, columns=periods) # Create a DataFrame equivalent to the MATLAB cell array.

    # Write the Excel sheet using pandas' optimal method.
    df.to_excel(writer, sheet_name=sheet_name, index_label='Costs in B€/y')
