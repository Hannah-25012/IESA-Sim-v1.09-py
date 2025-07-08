import numpy as np


def disp_energy_scarcity(dimensions, activities, technologies, iP):
    # Extract Parameters
    nAe = dimensions['nAe']
    nTb = dimensions['nTb']
    activities_names = activities['names']
    activities_energy = activities['energies']['names']
    energy_scarcity = activities['energies']['scarcity'][:, iP]
    activity_per_tech = technologies['balancers']['activities']
    activity_balances = technologies['balancers']['activity_balances']
    tech_categories = technologies['balancers']['categories']
    cap2act = technologies['balancers']['cap2acts']
    tech_stock = technologies['balancers']['stocks']['evolution'][:, iP]
    tech_use = technologies['balancers']['use']['yearly'][:, iP]

    # Begin the loop to quantify the differences between 
    energy_scarcity_new = np.zeros((nAe, 1))
    for iTb in range(nTb):
        if tech_categories[iTb] == 'Primary':
            # Identify the energy activity which has scarcity
            # Create boolean arrays to match the corresponding activity names
            iAe = np.array([name == activity_per_tech[iTb] for name in activities_energy])
            iA = np.array([name == activity_per_tech[iTb] for name in activities_names])
            
            # Quantify the scarcity
            activity_balance_val = activity_balances[iTb, iA][0]
            scarcity = tech_use[iTb] - tech_stock[iTb] * cap2act[iTb] * activity_balance_val
            scarcity = max(0, scarcity)
            
            # Add the scarcity to the account
            energy_scarcity_new[iAe, 0] = energy_scarcity_new[iAe, 0] + scarcity

    # Save Variables
    activities['energies']['scarcity'][:, iP] = energy_scarcity + energy_scarcity_new.flatten()
    
    
    return activities

