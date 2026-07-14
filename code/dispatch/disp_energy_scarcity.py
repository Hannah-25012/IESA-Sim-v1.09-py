# File to quantify the energy scarcity
import numpy as np

def disp_energy_scarcity(dimensions, activities, technologies, iP):

    # Extract Parameters
    nAe = dimensions['nAe']
    nTb = dimensions['nTb']
    energy_scarcity = activities.energies.scarcity[:, iP]
    tech_entities = technologies.balancers.entities
    tech_stock = technologies.balancers.stocks.evolution[:, iP]
    tech_use = technologies.balancers.use.yearly[:, iP]
    activity_balances = technologies.balancers.activity_balances
    cap2act = technologies.balancers.cap2acts

    # Identify Primary technologies whose main activity is an energy activity,
    # and each one's own main-activity balance coefficient, all at once
    # instead of a per-technology (nTb=550) Python loop.
    is_primary = np.array([tech.category == 'Primary' for tech in tech_entities])
    own_act_idx = np.array([tech.activity.idx if tech.activity is not None else -1 for tech in tech_entities])
    energy_idx = np.array([
        tech.activity.energy_idx if (tech.activity is not None and tech.activity.energy_idx is not None) else -1
        for tech in tech_entities
    ])
    valid = is_primary & (own_act_idx >= 0) & (energy_idx >= 0)

    activity_balance_val = np.zeros(nTb)
    activity_balance_val[valid] = activity_balances[np.arange(nTb)[valid], own_act_idx[valid]]

    scarcity = np.maximum(0.0, tech_use - tech_stock * cap2act * activity_balance_val)

    energy_scarcity_new = np.zeros(nAe)
    np.add.at(energy_scarcity_new, energy_idx[valid], scarcity[valid])

    # Save Variables
    activities.energies.scarcity[:, iP] = energy_scarcity + energy_scarcity_new

    return activities

