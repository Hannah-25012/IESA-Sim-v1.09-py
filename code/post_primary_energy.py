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
    # For each activity, flag if it is in the list of energy-related activities.
    coord_energy = np.array([act in activities_energy for act in activities_names])
    
    primary_energy = np.zeros(nEl)  # Preallocate primary energy vector 
    energy_exports = np.zeros(nAe)    # Preallocate energy exports vector 

    # Calculate the use of primary energies
    for iTb in range(nTb):
        # debugging - stop at first technology that's either primary or exports
        # if iTb > 279:
        #     breakpoint()
        # Process the technology
        # print(tech_categories[iTb])

        if tech_categories[iTb] == 'Primary':
            # Create a boolean mask for activities matching the current technology’s activity
            # and which have been flagged (coord_energy)
            coord_act = np.array([
                (activities_names[i] == activityPer_tech[iTb]) and coord_energy[i]
                for i in range(nA)
            ])
            if not np.any(coord_act):
                continue  # no matching activity found, skip to next technology
            
            # In MATLAB: activity_label(coord_act) returns a (presumably unique) string.
            # Here, we find all indices where coord_act is True and take the first match.
            matching_indices = np.where(coord_act)[0]
            selected_index = matching_indices[0]
            selected_activity_label = activity_label[selected_index]
            
            # Determine energy index: MATLAB uses strcmp(energy_labels, activity_label(coord_act))
            iE = np.array([el == selected_activity_label for el in energy_labels])
            
            # Sum the activity balances for the matching activity(ies)
            balance_sum = np.sum(activity_balances[iTb, coord_act])
            
            # Update primary energy: equivalent to
            # primary_energy(iE) = primary_energy(iE) + tech_use(iTb)*activity_balances(iTb,coord_act)
            primary_energy[iE] += tech_use[iTb] * balance_sum

        if tech_categories[iTb] == 'Exports':
            # For exports, select activities where the balance is negative and flagged in coord_energy
            coord_act = ((activity_balances[iTb, :] < 0) & coord_energy)
            if not np.any(coord_act):
                continue
            matching_indices = np.where(coord_act)[0]
            selected_index = matching_indices[0]
            
            # In MATLAB: coord_energyV = strcmp(activities_energy, activities(coord_act));
            # That is, compare the (unique) activity name (from activities where coord_act is True)
            # with each element in activities_energy.
            selected_activity_name = activities_names[selected_index]
            coord_energyV = np.array([ae == selected_activity_name for ae in activities_energy])
            
            # Get the corresponding activity label 
            selected_activity_label = activity_label[selected_index]
            # Determine energy index iE 
            iE = np.array([el == selected_activity_label for el in energy_labels])
            
            balance_sum = np.sum(activity_balances[iTb, coord_act])
            
            primary_energy[iE] += tech_use[iTb] * balance_sum
            energy_exports[coord_energyV] -= tech_use[iTb] * balance_sum

    # Adapt for the system imbalances
    # Note: Until here, everything seems fine. So the mistake must happen somewhere in this part.
    # It's probably again because tech_use for 466 and 467 is off. All the other numbers in tech_use seem to match. 
    actBalance = tech_use[:, None] * activity_balances  # broadcasting tech_use (nTb x 1) over nA columns
    # debugging
    # activity_balances_sum = np.sum(activity_balances, axis = 0)
    # print("sum of activity_balances for each activity:", activity_balances_sum)
    # numbers 70, 71, 74, 75, 76, 77, 80, etc. are off. activity_balances are correct, so this must come from tech_use. The numbers are very small.
    # this discrepancy is probably also the reason why numbers further down don't match. So it again comes back to tech_use.
    activity_total = np.sum(actBalance, axis=0)
    # debugging
    # print("sum of activity_total:", np.sum(activity_total))

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
    # Note: numbers 2 and 7 are slightly off
    results['primary'][:, iP] = primary_energy
    # Note: numbers 8, 15, 16, 25, 37, 40, 41, 43, 44, 48, 52, 54
    results['exports'][:, iP] = energy_exports

    return results
