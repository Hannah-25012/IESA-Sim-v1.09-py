# File to quantify the energy scarcity
import numpy as np

def disp_energy_scarcity(dimensions, activities, technologies, iP):

    # Extract Parameters
    nAe = dimensions['nAe']
    energy_scarcity = activities.energies.scarcity[:, iP]
    tech_entities = technologies.balancers.entities
    tech_stock = technologies.balancers.stocks.evolution[:, iP]
    tech_use = technologies.balancers.use.yearly[:, iP]

    # Begin the loop to quantify the differences between use and available energy
    energy_scarcity_new = np.zeros((nAe, 1))
    for tech in tech_entities:
        if tech.category == 'Primary' and tech.activity is not None:

            # Quantify the scarcity
            activity_balance_val = tech.activity_balances[tech.activity.idx]
            scarcity = tech_use[tech.idx] - tech_stock[tech.idx] * tech.cap2act * activity_balance_val
            scarcity = max(0, scarcity)

            # Add the scarcity to the account, if this technology's main activity is an energy activity
            if tech.activity.energy_idx is not None:
                energy_scarcity_new[tech.activity.energy_idx, 0] += scarcity

    # Save Variables
    activities.energies.scarcity[:, iP] = energy_scarcity + energy_scarcity_new.flatten()

    return activities

