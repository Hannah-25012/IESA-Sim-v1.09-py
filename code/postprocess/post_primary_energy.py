# File to generate the primary energy balance to report
import numpy as np

def post_primary_energy(dimensions, types, activities, technologies, results, iP):

    # Extract Parameters
    nEl = dimensions['nEl']
    nAe = dimensions['nAe']
    energy_labels = types.energy.labels
    activity_entities = activities.entities
    tech_entities = technologies.balancers.entities
    activity_balances = technologies.balancers.activity_balances
    tech_use = technologies.balancers.use.yearly[:, iP]

    # Energy-label -> position lookup, resolved once instead of re-scanning
    # energy_labels for every technology/activity.
    label_idx_by_name = {name: i for i, name in enumerate(energy_labels)}

    # Preallocate the primary energy vector
    coord_energy = np.array([a.is_energy for a in activity_entities]) # For each activity, flag if it is energy-related.
    primary_energy = np.zeros(nEl)  # Preallocate primary energy vector
    energy_exports = np.zeros(nAe)    # Preallocate energy exports vector

    # Calculate the use of primary energies
    for tech in tech_entities:
        iTb = tech.idx

        if tech.category == 'Primary':

            # The technology's own activity must be energy-related for this to apply
            act = tech.activity
            if act is None or not act.is_energy:
                continue  # no matching energy activity found, skip to next technology
            iE = label_idx_by_name[act.label]
            balance_sum = tech.activity_balances[act.idx] # Sum reduces to the technology's own activity balance
            primary_energy[iE] += tech_use[iTb] * balance_sum # Update primary energy: equivalent to primary_energy(iE) = primary_energy(iE) + tech_use(iTb)*activity_balances(iTb,coord_act)

        # 'XC Trade' (the electricity interconnectors of an Opt-structured database): net flow = import minus export, as in CBS
        # and as IESA-Opt books it (julia-backend _period_summary_payload). Without this the pinned trade (2022: -15 PJ, 2025: -51 PJ
        # in CBS) never reached primary energy and Sim showed only its surplus electricity (-7 / -8 PJ). The flow is the technology's
        # use in PJ; exports are the techs whose name says so.
        if tech.category == 'XC Trade':
            xc_act = tech.activity
            xc_label = xc_act.label if (xc_act is not None and xc_act.label in label_idx_by_name) else 'Electricity'
            if xc_label in label_idx_by_name:
                xc_sign = -1.0 if 'Export' in (tech.name or '') else 1.0
                primary_energy[label_idx_by_name[xc_label]] += xc_sign * tech_use[iTb]

        if tech.category == 'Exports':
            coord_act = (tech.activity_balances < 0) & coord_energy # For exports, select activities where the balance is negative and energy-related
            if not np.any(coord_act):
                continue
            matching_indices = np.where(coord_act)[0] # We find all indices where coord_act is True and take the first match.
            selected_activity = activity_entities[matching_indices[0]]
            iE = label_idx_by_name[selected_activity.label] # Determine energy index
            balance_sum = np.sum(tech.activity_balances[coord_act]) # Sum the activity balances for the matching activity(ies)
            primary_energy[iE] += tech_use[iTb] * balance_sum
            energy_exports[selected_activity.energy_idx] -= tech_use[iTb] * balance_sum

    # International bunkers (ships and aircraft serving international routes) are not part of
    # primary energy supply in CBS's energy balance - CBS deducts them from TPES. In a merged
    # (Opt-structured) database the bunker technologies consume ordinary fuels, so that fuel
    # would otherwise be counted as supply. Remove it by the energy label of the fuel consumed
    # (same technology rule as the final-energy table: subsector 'Bunker', 'International' in the
    # name, or the conventional airplane). Keep identical to julia-backend _period_summary_payload.
    for tech in tech_entities:
        name = tech.name or ""
        if not (tech.subsector == 'Bunker' or 'International' in name
                or (tech.sector == 'Transport' and 'Conventional Airplane' in name)):
            continue
        iTb = tech.idx
        for act in activity_entities:
            if act.energy_idx is None or not act.is_energy:
                continue
            bal = activity_balances[iTb, act.idx]
            if bal < 0:  # consumption of an energy carrier
                primary_energy[label_idx_by_name[act.label]] += tech_use[iTb] * bal

    # Adapt for the system imbalances
    actBalance =tech_use[:, None] * activity_balances  # broadcasting tech_use (nTb x 1) over nA columns
    activity_total = np.sum(actBalance, axis=0)

    for act in activity_entities:
        if act.energy_idx is None:
            continue
        iA = act.idx

        # Calculate the overproduced energy
        if activity_total[iA] > 0:
            iE = label_idx_by_name[act.label]
            primary_energy[iE] -= activity_total[iA]
            energy_exports[act.energy_idx] += activity_total[iA]

        # Calculate the underproduced energy (only if the label is not 'Electricity')
        if activity_total[iA] < 0 and ('Electricity' not in act.label):
            iE = label_idx_by_name[act.label]
            primary_energy[iE] -= activity_total[iA]

    # Save Variables
    results.primary[:, iP] = primary_energy
    results.exports[:, iP] = energy_exports

    return results
