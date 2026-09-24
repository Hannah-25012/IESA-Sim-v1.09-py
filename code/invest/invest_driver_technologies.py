# File to invest driver technologies to fill activity gaps
import numpy as np

def invest_driver_technologies(dimensions, activities, technologies,
                               tech_stock_exist, investment_potential, activity_gap, tech_choices, iP,
                               frozen_mix=False):

    # Extract Parameters
    nTb = dimensions['nTb']
    driver_activities = [a for a in activities.entities if a.is_driver]
    cap2act = technologies.balancers.cap2acts
    shedding_guarantee = technologies.balancers.shedding.guarantee
    tech_stock_max = technologies.balancers.stocks.max[:, iP]
    tech_choices_LCOP_order = technologies.balancers.choices_lcop_order[:, iP]

    # Calculate the shedding adjustment
    coord_shedding_tech = shedding_guarantee > 0
    shedding_adjustment = np.ones(nTb)
    shedding_adjustment[coord_shedding_tech] = shedding_guarantee[coord_shedding_tech]

    # For every activity, add sufficient capacity
    new_investments = np.zeros((nTb,1))

    for activity in driver_activities:

        tech_coord = np.array([t.idx for t in activity.technologies], dtype=int)
        nT = len(tech_coord)
        cap2act_temp = cap2act[tech_coord]
        shedding_adjustment_temp = shedding_adjustment[tech_coord]

        # Check if the activity has technologies
        if nT > 0:
            # Check investment potential constraint
            choices_investments = (activity_gap[activity.idx] * tech_choices[tech_coord] /
                                    cap2act_temp / shedding_adjustment_temp)
            valid_investments = np.maximum(
                np.minimum(investment_potential[tech_coord], choices_investments), 0
            )

            # Check sufficiency
            remaining_gap = activity_gap[activity.idx] - np.sum(valid_investments * cap2act_temp * shedding_adjustment_temp)
            fill_investments = np.zeros((nT,1))  # Preallocate
            frozen_fill = np.zeros(nT)

            if remaining_gap > 0:
                tech_room = (tech_stock_max[tech_coord] - tech_stock_exist[tech_coord] - valid_investments)

                # Frozen-mix scenario (see invest_tech_choices_per_act.py,
                # invest_power_technologies.py, and
                # invest_energy_technologies.py for the same rule on the
                # other investment paths): before the LCOP-ordered/max-room
                # fallback below - which fills the remaining gap by cost
                # ranking or by dumping onto whichever technology has the
                # most room, ignoring tech_choices/existing shares entirely -
                # water-fill as much of it as possible across technologies
                # that already have existing stock, proportional to that
                # stock's own share. Iterative because a technology can hit
                # its own room before its "fair share" of the gap is used
                # up, in which case the leftover re-splits across the
                # technologies that still have room - by construction this
                # only stops early with gap left over once EVERY technology
                # with existing stock has had its room fully consumed, so
                # the fallback below then only ever has zero-stock (nothing
                # to freeze) technologies left to pick from anyway.
                if frozen_mix:
                    stock_existing = np.asarray(tech_stock_exist[tech_coord]).reshape(-1)
                    tech_room_flat = np.asarray(tech_room).reshape(-1)
                    room_activity = tech_room_flat * cap2act_temp * shedding_adjustment_temp
                    active = (tech_room_flat > 0) & (stock_existing > 0)
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
                        frozen_fill = np.where(alloc_activity > 0,
                                                alloc_activity / (cap2act_temp * shedding_adjustment_temp), 0.0)
                    tech_room = tech_room_flat - frozen_fill
                    remaining_gap = gap_left
                    if remaining_gap < 1e-6:
                        remaining_gap = 0

                cand_availability = tech_room > 0
                order_search = tech_choices_LCOP_order[tech_coord]
                iT_search = 1

                while remaining_gap > 0:
                    iT = np.where(order_search == iT_search)[0]
                    if cand_availability[iT]:
                        fill_investments[iT] = min(
                            remaining_gap / cap2act_temp[iT] / shedding_adjustment_temp[iT],
                            tech_room[iT]
                        )
                    remaining_gap -= fill_investments[iT] * cap2act_temp[iT] * shedding_adjustment_temp[iT]

                    if remaining_gap < 1e-6:
                        remaining_gap = 0
                    iT_search += 1

                    if iT_search > nT and remaining_gap > 0:
                        tech_room_new = (tech_stock_max[tech_coord] -
                                         tech_stock_exist[tech_coord] - valid_investments - fill_investments - frozen_fill)
                        indices = np.where(tech_room_new == np.max(tech_room_new))[0]
                        iT = indices[0] if len(indices) > 0 else None
                        fill_investments[iT, 0] += remaining_gap[0, 0] / cap2act_temp[iT] / shedding_adjustment_temp[iT]
                        remaining_gap = 0

            # Merge the frozen-mix water-fill pass (if any) in now - only
            # after the LCOP-ordered/max-room fallback above has run, since
            # that fallback writes into fill_investments directly rather than
            # adding to it (see the comment where frozen_fill is computed).
            fill_investments[:, 0] += frozen_fill

            valid_investments = valid_investments.reshape(-1, 1)
            updated_investments = valid_investments + fill_investments
            new_investments[tech_coord, 0] = np.maximum(updated_investments.flatten(), 0)

    # Adjust investments in shedding technologies
    new_investments[coord_shedding_tech] /= shedding_guarantee[coord_shedding_tech].reshape(-1, 1)

    # Save variables
    tech_stock_exist = tech_stock_exist.reshape(-1, 1)
    tech_stock_new = tech_stock_exist + new_investments

    return tech_stock_new, new_investments
