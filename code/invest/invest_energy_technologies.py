# File to invest in energy technologies
import numpy as np
from invest_energy_sufficiency import invest_energy_sufficiency
from invest_investment_potential import DEFAULT_STOCK_DEPLOY_FRACTION

def invest_energy_technologies(dimensions, activities, technologies, techstock_exist,
                                investment_potential, tech_choices, iP, frozen_mix=False):
    # Extract Parameters
    nI = 3  # Number of investment iterations
    nA = dimensions['nA']
    nAe = dimensions['nAe']
    nTb = dimensions['nTb']
    energy_activities = [a for a in activities.entities if a.is_energy]
    cap2act = technologies.balancers.cap2acts
    shedding_guarantee = technologies.balancers.shedding.guarantee
    techstock_max = technologies.balancers.stocks.max[:, iP]
    # Same default pacing rule as invest_investment_potential.py's Tier 1
    # (investment_potential): a technology with no real stock_deploy set
    # falls back to a fraction of its own stock_max instead of unlimited
    # per-period headroom - computed once here so both fallback tiers below
    # can't bypass it the way they used to bypass a real stock_deploy value.
    stock_deploy_raw = technologies.balancers.stocks.deploy
    stock_deploy = np.where(stock_deploy_raw > 0, stock_deploy_raw,
                             DEFAULT_STOCK_DEPLOY_FRACTION * techstock_max)
    tech_choices_lcop_order = technologies.balancers.choices_lcop_order[:, iP]
    activity_balances = technologies.balancers.activity_balances

    # Calculate the shedding adjustment
    coord_shedding_tech = shedding_guarantee > 0
    shedding_adjustment = np.ones(nTb)
    shedding_adjustment[coord_shedding_tech] = shedding_guarantee[coord_shedding_tech]

    # For every activity add sufficient capacity
    investments = np.zeros((nTb,1))
    techstock_exist_check = techstock_exist.copy()

    # debugging: apply rounding to 14 decimals (same is done in matlab)
    precision = 14

    for iI in range(nI):
        # Determine the energy gap for current stock
        energy_gap = invest_energy_sufficiency(dimensions, activities, technologies,
                                               techstock_exist_check, False, iP)
        energy_gap = np.round(energy_gap, decimals=precision)

        # Determined required investments in the iteration
        iter_investments = np.zeros((nTb,1))
        for activity in energy_activities:
            if not activity.is_electricity and energy_gap[activity.idx] > 0:

                tech_coord = np.array([t.idx for t in activity.technologies], dtype=int)
                nT = len(tech_coord)
                cap2act_temp = cap2act[tech_coord]
                shedding_adjustment_temp = shedding_adjustment[tech_coord]

                # Check if the activity has technologies to invest in
                if nT > 0:
                    # Check maximum stock constraint
                    choices_investments = (energy_gap[activity.idx] * tech_choices[tech_coord] /
                                           cap2act_temp / shedding_adjustment_temp)
                    choices_investments = np.round(choices_investments, decimals=precision)
                    valid_investments = np.maximum(
                        np.minimum(investment_potential[tech_coord], choices_investments), 0
                    )
                    valid_investments = np.round(valid_investments, decimals=precision)

                    # Check sufficiency
                    remaining_gap = energy_gap[activity.idx] - \
                                    np.sum(valid_investments * cap2act_temp * shedding_adjustment_temp)
                    remaining_gap = np.round(remaining_gap, decimals=precision)
                    fill_investments = np.zeros((nT,1))
                    other_investments = np.zeros((nT,1))
                    frozen_fill = np.zeros(nT)

                    if remaining_gap > 0:
                        tech_room = (techstock_max[tech_coord].flatten() - techstock_exist[tech_coord].flatten() -
                                     valid_investments.flatten())
                        # BUGFIX: this "fill the remaining gap" tier previously
                        # ignored stock_deploy entirely, bounded only by
                        # techstock_max - so a technology with stock_deploy>0
                        # (a real per-period build-rate cap, respected by
                        # valid_investments above via investment_potential)
                        # could still be force-invested straight up to its
                        # TOTAL ceiling the moment a scarcity gap exceeded
                        # that pace, defeating the cap's whole purpose exactly
                        # when it matters most (a sudden gap, e.g. a retiring
                        # technology). Layer the same per-period pace limit on
                        # top of techstock_max here: sd_room tracks how much
                        # of THIS period's stock_deploy allowance is left
                        # after what this function has already committed
                        # (investments, from prior iI passes) plus this
                        # iteration's own valid_investments tier. `stock_deploy`
                        # here is already the effective value from the top of
                        # this file - a real per-technology figure where set,
                        # else the DEFAULT_STOCK_DEPLOY_FRACTION default - so
                        # sd_room is only +inf if techstock_max itself is 0
                        # (room_to_invest is 0 there too, so it's moot).
                        sd_tech = stock_deploy[tech_coord]
                        sd_used = investments[tech_coord, 0] + valid_investments.flatten()
                        sd_room = np.where(sd_tech > 0, np.maximum(0.0, sd_tech - sd_used), np.inf)
                        tech_room = np.minimum(tech_room, sd_room)
                        tech_room = np.round(tech_room, decimals=precision)

                        # Frozen-mix scenario (see invest_tech_choices_per_act.py
                        # and invest_power_technologies.py for the same rule on
                        # the other investment paths): before the LCOP-ordered/
                        # round-robin fallback below - which fills the gap by
                        # cost ranking, ignoring tech_choices/existing shares
                        # entirely - water-fill as much of the remaining gap as
                        # possible across technologies that already have
                        # existing stock, proportional to that stock's own
                        # share. This is the same rule as valid_investments
                        # above, extended past tier 1's own investment_potential
                        # cap into whatever stock_deploy-limited room is left.
                        # Iterative because a technology can hit its own room
                        # before its "fair share" of the gap is used up, in
                        # which case the leftover re-splits across the
                        # technologies that still have room. A technology with
                        # zero existing stock has nothing to freeze, so it's
                        # excluded here and left entirely to the LCOP-ordered
                        # fallback below - and so is whatever gap an existing
                        # producer's own stock_deploy/DEFAULT_STOCK_DEPLOY_FRACTION
                        # pace cap won't let it absorb, even though it does have
                        # stock (e.g. Gas Reforming's real per-period Hydrogen
                        # growth limit vs. faster demand growth). Deliberate,
                        # not a gap to close: the build-rate cap stays real
                        # under frozen_mix too, so "nothing changes" still
                        # means "nothing changes FASTER than physically
                        # plausible," not "unlimited replacement capacity."
                        # tech_room/remaining_gap are reduced by
                        # whatever this pass places, so the unchanged fallback
                        # code below only ever sees what's left over - and
                        # frozen_fill itself is merged into fill_investments
                        # only after that fallback runs (its own while loop
                        # OVERWRITES fill_investments[selected_index] rather
                        # than adding to it, so merging any earlier would lose
                        # this pass's contribution the moment the same index
                        # got touched again).
                        if frozen_mix:
                            stock_existing = np.asarray(techstock_exist[tech_coord]).reshape(-1)
                            room_activity = tech_room * cap2act_temp * shedding_adjustment_temp
                            active = (tech_room > 0) & (stock_existing > 0)
                            alloc_activity = np.zeros(nT)
                            gap_left = float(remaining_gap)
                            while gap_left > 1e-9 and active.any():
                                active_stock = np.where(active, stock_existing, 0.0)
                                stock_sum = active_stock.sum()
                                if stock_sum <= 0:
                                    break
                                shares = active_stock / stock_sum
                                headroom = room_activity - alloc_activity
                                give = np.clip(np.minimum(gap_left * shares, headroom), 0, None)
                                if give.sum() <= 1e-12:
                                    break
                                alloc_activity += give
                                gap_left -= give.sum()
                                active = active & ((room_activity - alloc_activity) > 1e-9)

                            with np.errstate(divide='ignore', invalid='ignore'):
                                frozen_fill = np.where(
                                    alloc_activity > 0,
                                    alloc_activity / (cap2act_temp * shedding_adjustment_temp), 0.0
                                )
                            frozen_fill = np.round(frozen_fill, decimals=precision)
                            tech_room = np.round(tech_room - frozen_fill, decimals=precision)
                            remaining_gap = np.round(gap_left, decimals=precision)
                            if remaining_gap < 1e-6:
                                remaining_gap = 0

                        cand_availability = tech_room > 0
                        order_search = tech_choices_lcop_order[tech_coord]
                        iT_search = 1

                        while remaining_gap > 0:
                            iT = np.where(order_search == iT_search)[0]
                            if len(iT) > 0 and cand_availability[iT].any():
                                selected_index = iT[0]
                                selected_index = int(selected_index)
                                remaining_gap_ratio = remaining_gap / (cap2act_temp[selected_index].item() * shedding_adjustment_temp[selected_index].item())
                                available_tech_room = tech_room[selected_index].item()
                                fill_investments[selected_index] = min(remaining_gap_ratio, available_tech_room)

                            remaining_gap -= (fill_investments[iT[0]] * cap2act_temp[iT[0]] *
                                              shedding_adjustment_temp[iT[0]])
                            remaining_gap = np.round(remaining_gap, decimals=precision)

                            if remaining_gap < 1e-6:
                                remaining_gap = 0
                            iT_search += 1

                            if iT_search > nT and remaining_gap > 0:

                                # Check for other technologies that were not chosen by agents
                                tech_room_others = (techstock_max[tech_coord] - techstock_exist[tech_coord].ravel() -
                                                    valid_investments.ravel() - fill_investments.ravel() - frozen_fill)
                                # Same stock_deploy layering as the main fill
                                # loop above - this fallback tier would
                                # otherwise bypass the pace cap too. frozen_fill
                                # (the frozen-mix water-fill pass above) is
                                # included in both room and pace accounting here
                                # since it hasn't been merged into
                                # fill_investments yet - see the comment where
                                # frozen_fill is computed.
                                sd_used_others = (investments[tech_coord, 0] + valid_investments.ravel() +
                                                   fill_investments.ravel() + frozen_fill)
                                sd_room_others = np.where(sd_tech > 0, np.maximum(0.0, sd_tech - sd_used_others), np.inf)
                                tech_room_others = np.minimum(tech_room_others, sd_room_others)
                                if np.sum(tech_room_others) > 0:
                                    for iT in range(nT):
                                        other_investments[iT] = max(
                                            min(tech_room_others[iT] * cap2act_temp[iT], remaining_gap), 0
                                        ) / cap2act_temp[iT] / shedding_adjustment_temp[iT]
                                        remaining_gap -= (other_investments[iT] * cap2act_temp[iT] *
                                                          shedding_adjustment_temp[iT])

                                # If still gap remains, display warning
                                if remaining_gap > 0:
                                    if iI == nI - 1:
                                        print(f"!!!!Warning: Not enough capacity to satisfy the demand for {activity.name}. Remaining gap: {remaining_gap.item():.2f} UoA")
                                    remaining_gap = 0

                    # Merge the frozen-mix water-fill pass (if any) in now -
                    # only after the LCOP-ordered/round-robin fallback above
                    # has run, since that fallback's own while loop OVERWRITES
                    # fill_investments[selected_index] rather than adding to
                    # it (see the comment where frozen_fill is computed).
                    fill_investments[:, 0] += frozen_fill

                    valid_investments = np.round(valid_investments.reshape(-1, 1), decimals=precision)
                    fill_investments = np.round(fill_investments.reshape(-1, 1), decimals=precision)
                    other_investments = np.round(other_investments.reshape(-1, 1), decimals=precision)
                    updated_investments = valid_investments + fill_investments + other_investments
                    iter_investments[tech_coord, 0] = np.maximum(np.round(updated_investments.flatten(), decimals=precision), 0)

        # Updated existing stock to determine gaps
        techstock_exist_check += iter_investments
        techstock_exist_check = np.round(techstock_exist_check, decimals=precision)
        investments += iter_investments
        investments = np.round(investments, decimals=precision)

    # Define investments
    techstock_new = techstock_exist + investments
    techstock_new = np.round(techstock_new, decimals=precision)

    # Check the gaps and report them
    energy_gap = invest_energy_sufficiency(dimensions, activities, technologies, techstock_new, False, iP)
    energy_gap = np.round(energy_gap, decimals=precision)

    # Check which energy forms present scarcity
    energy_scarcity_bin = np.zeros((nA, 1), dtype=bool)
    coord_gap = (energy_gap > 1e-12).reshape(-1, 1)
    is_electricity = np.array([a.is_electricity for a in activities.entities]).reshape(-1, 1)
    coord_nonE = (~is_electricity)
    logical_indices = np.logical_and(coord_gap, coord_nonE)
    energy_scarcity_bin[logical_indices] = True

    # Facilitate the resources for this energy to be satisfied
    tech_preference = tech_choices.copy()
    for activity in activities.entities:
        if energy_scarcity_bin[activity.idx]:

            # Identify possible technologies - producers of this activity
            # only. activity.technologies (entities.py's link_entities) is
            # built purely from each technology's own main "activity_name"
            # field, with no regard for whether it actually produces or
            # consumes that activity - an Exports-category technology (e.g.
            # "Exported Hydrogen") lists the very activity it exports as its
            # own main activity too, with a NEGATIVE activity_balances
            # coefficient (it removes the activity from the system, it
            # doesn't supply it). Filtering to a positive coefficient here
            # keeps this scarcity fallback from ever "investing" in a sink to
            # fix a supply shortfall - confirmed: without this filter, a
            # persistent Hydrogen HD gap (its real producer technologies
            # never became cost-competitive) got "filled" every period by
            # force-investing in Hyd03_02 (Exported Hydrogen) instead, which
            # did nothing to relieve the shortage - the same unmet gap then
            # recurred (and compounded, via the techstock_max overwrite this
            # fix also removes below) every subsequent period, ratcheting
            # that one technology's own stock/dispatched use into the
            # thousands with no physical production ever behind it.
            tech_coord_all = np.array([t.idx for t in activity.technologies], dtype=int)
            tech_coord = tech_coord_all[activity_balances[tech_coord_all, activity.idx] > 0]

            # An activity can have an energy gap but no producing technology
            # at all to fill it - either no technology names it as their main
            # activity in the first place (e.g. an IESA-Sim-side activity
            # name with no matching IESA-Opt-side technology after a
            # merged-database load - see dbcompare-backend's /unify - such as
            # "Natural Gas"/"Hydrogen" vs. the technology-bearing "Natural Gas
            # HD"/"Natural Gas LD" naming), or every technology that does is
            # actually a consumer/exporter of it (just filtered out above) -
            # skip it like the analogous nT == 0 case in
            # invest_tech_choices_per_act.py, instead of crashing on np.max
            # of an empty array.
            if tech_coord.size == 0:
                print(f"--****There is no producing technology to cover the energy gap for activity: {activity.name}")
                continue

            # Frozen-mix scenario (see invest_tech_choices_per_act.py and
            # invest_power_technologies.py for the same rule on the other two
            # investment paths): split the unmet gap across ALL current
            # producers by their existing stock share, instead of dumping the
            # whole thing into a single highest-tech_choices winner below - a
            # scarcity gap gets shared out like-for-like too, not handed to
            # whichever technology currently ranks first. A gap for an
            # activity with no current producer stock (nothing built yet)
            # still falls through to the normal winner-take-all pick, since
            # there's no existing mix to freeze there.
            if frozen_mix:
                stock_for_shares = np.asarray(techstock_new[tech_coord]).reshape(-1)
                stock_total = stock_for_shares.sum()
            else:
                stock_total = 0.0

            if frozen_mix and stock_total > 0:
                shares = stock_for_shares / stock_total
                for i, t_idx in enumerate(tech_coord):
                    add = energy_gap[activity.idx] * shares[i] / cap2act[t_idx]
                    investments[t_idx] += add
                    techstock_new[t_idx] += add

            else:
                # Preferences of thoses technologies
                options_preferences = tech_preference[tech_coord]
                max_value = np.max(options_preferences)
                options_sel = np.where(options_preferences == max_value)[0][0]

                # Identify which technology was chosen
                choice_coord = tech_coord[options_sel]

                # Fill investments and stocks. techstock_max is deliberately left
                # untouched here (previously overwritten to match whatever was
                # just forced in) - that turned a real, exogenous capacity
                # ceiling into a self-referential bookkeeping artifact that only
                # ever grew, and fed straight into invest_techstocks_def.py's own
                # "guaranteed availability" forcing for Primary/Exports/
                # Undispatched-style technologies (tech_stock = tech_stock_max)
                # on any later period. If this technology's own real max still
                # constrains it, invest_techstocks_def.py's normal min/max clamp
                # brings the forced investment back down to that genuine ceiling
                # next period - the same honest "can't fully meet this" outcome
                # the warning above already reports, not a silently expanded one.
                investments[choice_coord] += energy_gap[activity.idx] / cap2act[choice_coord]
                techstock_new[choice_coord] += energy_gap[activity.idx] / cap2act[choice_coord]

    # Calculate energy scarcity
    energy_scarcity_add = np.zeros((nAe,1))
    for iAe, activity in enumerate(energy_activities):
        if energy_scarcity_bin[activity.idx]:
            energy_scarcity_add[iAe] = energy_gap[activity.idx][0]

    # Save variables
    energy_scarcity = activities.energies.scarcity[:, iP].copy()
    activities.energies.scarcity[:, iP] = energy_scarcity + energy_scarcity_add.flatten()
    technologies.balancers.stocks.max[:, iP] = techstock_max

    return technologies, activities, techstock_new, investments
