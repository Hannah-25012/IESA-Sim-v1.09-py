# File to generate the primary energy balance to report
import numpy as np

def post_primary_energy(dimensions, types, activities, technologies, results, iP):

    # Extract Parameters
    nA = dimensions['nA']
    nAe = dimensions['nAe']
    nEl = dimensions['nEl']
    nTb = dimensions['nTb']
    energy_labels = types['energy']['labels']
    activities_names = activities['names']        
    activity_label = activities['labels']           
    activities_energy = activities['energies']['names']  
    activityPer_tech = technologies['balancers']['activities']  
    tech_categories = technologies['balancers']['categories']
    activity_balances = technologies['balancers']['activity_balances']
    tech_use = technologies['balancers']['use']['yearly'][:, iP] 

    # Preallocate the primary energy vector
    coord_energy = np.array([act in activities_energy for act in activities_names]) # For each activity, flag if it is in the list of energy-related activities.
    primary_energy = np.zeros(nEl)  # Preallocate primary energy vector 
    energy_exports = np.zeros(nAe)    # Preallocate energy exports vector 

    # Calculate the use of primary energies
    for iTb in range(nTb):
        if tech_categories[iTb] == 'Primary':
            coord_act = np.array([
                (activities_names[i] == activityPer_tech[iTb]) and coord_energy[i]
                for i in range(nA)
            ]) # Create a boolean mask for activities matching the current technology’s activity and which have been flagged (coord_energy)
            if not np.any(coord_act):
                continue  # no matching activity found, skip to next technology
            matching_indices = np.where(coord_act)[0] # In MATLAB: activity_label(coord_act) returns a (presumably unique) string. Here, we find all indices where coord_act is True and take the first match.
            selected_index = matching_indices[0]
            selected_activity_label = activity_label[selected_index]
            iE = np.array([el == selected_activity_label for el in energy_labels]) # Determine energy index: MATLAB uses strcmp(energy_labels, activity_label(coord_act))
            balance_sum = np.sum(activity_balances[iTb, coord_act]) # Sum the activity balances for the matching activity(ies)
            primary_energy[iE] += tech_use[iTb] * balance_sum # Update primary energy: equivalent to primary_energy(iE) = primary_energy(iE) + tech_use(iTb)*activity_balances(iTb,coord_act)

        if tech_categories[iTb] == 'Exports':
            coord_act = ((activity_balances[iTb, :] < 0) & coord_energy) # For exports, select activities where the balance is negative and flagged in coord_energy
            if not np.any(coord_act):
                continue
            matching_indices = np.where(coord_act)[0]
            selected_index = matching_indices[0]
            selected_activity_name = activities_names[selected_index]
            coord_energyV = np.array([ae == selected_activity_name for ae in activities_energy])
            selected_activity_label = activity_label[selected_index] # Get the corresponding activity label 
            iE = np.array([el == selected_activity_label for el in energy_labels]) # Determine energy index iE 
            balance_sum = np.sum(activity_balances[iTb, coord_act])
            primary_energy[iE] += tech_use[iTb] * balance_sum
            energy_exports[coord_energyV] -= tech_use[iTb] * balance_sum

    # Adapt for the system imbalances
    actBalance = tech_use[:, None] * activity_balances  # broadcasting tech_use (nTb x 1) over nA columns
    activity_total = np.sum(actBalance, axis=0)

    for iA in range(nA):
        coord_energyV = np.array([ae == activities_names[iA] for ae in activities_energy])
        
        # Calculate the overproduced energy
        if np.any(coord_energyV) and activity_total[iA] > 0:
            iE = np.array([el == activity_label[iA] for el in energy_labels])
            primary_energy[iE] -= activity_total[iA]
            energy_exports[coord_energyV] += activity_total[iA]
        
        # Calculate the underproduced energy (only if the label is not 'Electricity')
        if np.any(coord_energyV) and activity_total[iA] < 0 and ('Electricity' not in activity_label[iA]):
            iE = np.array([el == activity_label[iA] for el in energy_labels])
            primary_energy[iE] -= activity_total[iA]

    # Save Variables
    results['primary'][:, iP] = primary_energy
    results['exports'][:, iP] = energy_exports

    return results
