# File to dispatch generators
import numpy as np

def disp_power_generators(gen_vom, gen_balance_hourly, gen_availability_hourly,
                          gen_xc_costs_hourly, gen_per_elec, elec_demand_hourly,
                          prices_hourly, voll,
                          xc_to_idx, xc_from_actidx, xc_from_elecidx, xc_supply, xc_efficiencies, xc_vom, elec_is_nl,
                          shed_per_elec, shed_maxdemand_hourly, shed_mindemand_hourly, shed_vom, gen_is_voll):

# ========== Explanation ===============================================
# This function presents a merit order approach to determine prices per
# node taking into account only generation and load per hour.
# Interconnectors, shedding, and batteries are dispatched in other files.
#    Input:
#          1) gen_marginal_costs_hourly: float, matrix of houry marginal
#             costs per generator (nH,nG)
#          2) gen_availability_hourly: float, matrix of hourly
#             availabilities per generator (nH,nG)
#          3) elec_demand_hourly: float, matrix of hourly demand per
#             electricity activity (nH,nAk)
#          4) gen_per_elec: logical, matrix of generators installed per
#             electricity activity (nG,nAk)
#          5) shed_per_elec/shed_maxdemand_hourly/shed_mindemand_hourly/
#             shed_vom: shedding technologies' own node attachment, hourly
#             curtailable demand band, and VOM cost - used the same way as
#             the interconnector arguments below, as an alternate relief
#             source for a node whose own generator fleet can't meet demand
#    Output:
#          1) gen_use_hourly: float, matrix of hourly use per generator
#             (nH,nG)
#          2) elec_prices_hourly: float, matrix of hourly prices of the
#             electricity activities (nH,nAk)
#
# All nH hours are solved at once with array ops instead of a per-hour
# Python loop (the previous per-hour merit order was the single biggest
# CPU cost in the whole simulation). The only remaining Python loop is
# over electricity nodes (nAk), which is small.

    # Extract dimensions
    nG = gen_per_elec.shape[0]  # Number of generators
    nAk = gen_per_elec.shape[1]  # Number of electricity activities
    nH = gen_xc_costs_hourly.shape[0]  # Number of hours

    # Obtain the hourly marginal costs of the generator (batched dot product
    # of gen_balance_hourly (nH,nA,nG) with prices_hourly (nH,nA), one result
    # per generator, instead of a Python loop over generators)
    fuel_cost_component = np.einsum('han,ha->hn', gen_balance_hourly, prices_hourly)
    gen_marginal_costs_hourly = gen_vom[None, :] - fuel_cost_component + gen_xc_costs_hourly

    # For each electricity node, obtain the merit order curve, the marginal
    # generator, the resulting dispatch, and the marginal price - for all
    # hours simultaneously.
    gen_use_hourly = np.zeros((nH, nG))
    elec_prices_hourly = np.zeros((nH, nAk))
    # Local generation capacity minus demand, per node per hour - positive
    # means this node's own generators have spare capacity beyond its own
    # demand that hour, negative means they fall short. disp_power_interconnectors.py
    # uses this to bound how much an interconnector actually delivers (not
    # more than the receiving node needs, not more than the sending node
    # can actually spare), instead of dispatching its full nominal capacity
    # whenever merely profitable to use at all.
    net_available_hourly = np.zeros((nH, nAk))
    hours = np.arange(nH)

    # Per-node values needed by the relief/VOLL pass below, stashed here so
    # that pass can run only after EVERY node's own net_available_hourly is
    # known - seeing a node's relief source's OWN net_available_hourly
    # requires that source node to have already run its local merit order,
    # which a single combined loop can't guarantee regardless of iteration
    # order (the "from" node of an interconnector/transformer isn't
    # necessarily processed before the "to" node it's relieving).
    voll_true_by_node = np.zeros((nH, nAk), dtype=bool)
    marginal_cost_by_node = np.zeros((nH, nAk))

    # === Pass 1: local merit-order dispatch, per node =======================
    # Relief/VOLL never changes which or how much a LOCAL generator itself
    # runs - only the price charged for any residual gap (see pass 2) - so
    # gen_use_hourly and net_available_hourly are fully determined here.
    for iAk in range(nAk):

        # Define the problem to solve
        mask_gens = gen_per_elec[:, iAk].astype(bool)
        demand = elec_demand_hourly[:, iAk]
        gen_available = gen_availability_hourly * mask_gens[None, :]
        gen_cost = gen_marginal_costs_hourly * mask_gens[None, :]

        # Obtain merit order (independently per hour)
        MOC_order = np.argsort(gen_cost, axis=1, kind='stable')
        MOC_volume = np.cumsum(np.take_along_axis(gen_available, MOC_order, axis=1), axis=1)

        # Obtain marginal generator, volume, and price (independently per hour)
        meets_demand = MOC_volume >= demand[:, None]
        any_meets = meets_demand.any(axis=1)
        voll_true = ~any_meets
        MOC_last_gen = np.where(any_meets, meets_demand.argmax(axis=1), nG - 1)  # All generators online when voll_true
        MOC_volume_at_last = MOC_volume[hours, MOC_last_gen]
        MOC_excess = np.where(voll_true, 0.0, MOC_volume_at_last - demand)

        # net_available_hourly feeds disp_power_interconnectors.py's own
        # to_gap = max(-net_available_hourly, 0) - "does this node need real
        # relief". It must NOT be MOC_volume_at_last - demand (the full-
        # merit-order version above): when a VOLL/undispatched placeholder
        # (e.g. PNL_UD, PNL_UD_LV) sits in this node's own merit order, THAT
        # quantity is just the unused excess capacity of whichever generator
        # ends up marginal - when VOLL is marginal (its huge nameplate means
        # it usually is), that excess is VOLL's own near-total unused
        # headroom, not a real supply/demand balance, and it's completely
        # unrelated to how much VOLL actually had to dispatch that hour.
        # Confirmed live: subtracting VOLL's actual gen_use_hourly from this
        # value left it completely unchanged (159 PJ/year of real VOLL
        # dispatch, zero visible effect) - different quantities, not an
        # algebraic relationship. The fix is a SEPARATE real-generators-only
        # merit order: exclude every VOLL-type generator up front, so a node
        # that can't be served by its own real fleet shows a genuine,
        # negative shortfall here - exactly the signal a transformer/
        # interconnector needs to know it's actually needed. gen_use_hourly/
        # voll_true_by_node/marginal_cost_by_node below are deliberately left
        # on the FULL merit order (VOLL included) - actual dispatch and
        # pricing still correctly fall back to VOLL if real generation plus
        # relief isn't enough; only this one signal changes.
        real_mask = mask_gens & ~gen_is_voll
        gen_available_real = gen_availability_hourly * real_mask[None, :]
        gen_cost_real = gen_marginal_costs_hourly * real_mask[None, :]
        MOC_order_real = np.argsort(gen_cost_real, axis=1, kind='stable')
        MOC_volume_real = np.cumsum(np.take_along_axis(gen_available_real, MOC_order_real, axis=1), axis=1)
        meets_demand_real = MOC_volume_real >= demand[:, None]
        any_meets_real = meets_demand_real.any(axis=1)
        MOC_last_gen_real = np.where(any_meets_real, meets_demand_real.argmax(axis=1), nG - 1)
        MOC_volume_at_last_real = MOC_volume_real[hours, MOC_last_gen_real]
        net_available_hourly[:, iAk] = MOC_volume_at_last_real - demand

        iG_marginal = MOC_order[hours, MOC_last_gen]

        # Save dispatch parameters - final pricing deferred to pass 2.
        voll_true_by_node[:, iAk] = voll_true
        marginal_cost_by_node[:, iAk] = gen_cost[hours, iG_marginal]

        # Fill dispatch: set all online generators to full availability, then
        # subtract excess from the marginal generator.
        # BUGFIX: generators belonging to OTHER nodes were zeroed out above (not
        # excluded), so with cost 0 they sort to the front of the merit order and
        # can be marked "online" here too even though they contribute nothing -
        # writing them would clobber that generator's real dispatch from its own
        # node's turn. Restrict writes to generators that actually belong to this node.
        rank = np.argsort(MOC_order, axis=1)  # inverse permutation: rank[h, g] = position of g in hour h's merit order
        online = rank <= MOC_last_gen[:, None]
        gen_use_hourly[:, mask_gens] = np.where(online[:, mask_gens], gen_available[:, mask_gens], 0.0)

        marginal_is_own = mask_gens[iG_marginal]
        gen_use_hourly[hours[marginal_is_own], iG_marginal[marginal_is_own]] -= MOC_excess[marginal_is_own]

    # === Pass 2: relief/VOLL resolution, per node ============================
    # Every node's own net_available_hourly is now known regardless of
    # processing order, so a relief source's real spare capacity can be
    # checked no matter which side of it was dispatched first above.
    for iAk in range(nAk):
        voll_true = voll_true_by_node[:, iAk]
        marginal_cost = marginal_cost_by_node[:, iAk]
        demand = elec_demand_hourly[:, iAk]
        MOC_volume_at_last = demand + net_available_hourly[:, iAk]

        # Before falling back to VOLL: a node whose own generator fleet can't
        # meet demand may still have a connected interconnector/transformer,
        # or a shedding technology able to curtail some of its own demand,
        # able to close the gap - this only ever matters in an hour that's
        # already voll_true, so it can never make a normally-clearing hour's
        # price worse or change which generator sets the marginal price
        # there. Real generator dispatch (gen_use_hourly above) is untouched
        # either way - in a genuine shortage every available generator still
        # runs at full output regardless of whether the residual gap gets
        # priced at VOLL or at a relief source's own cost; each relief
        # source's own actual flow/curtailment is decided afterwards by
        # disp_power_interconnectors/disp_power_shedding, using whichever
        # price this resolves to - so there is no double-booking of the same
        # relief between this price-only check and that later, independent
        # dispatch decision (same reasoning as the pre-existing interconnector
        # check this shedding check is modeled on).
        #
        # Scoped to the Netherlands' own nodes (elec_is_nl) - every other
        # country node's own fleet is IESA-Sim's auxiliary representation of
        # a neighboring market, not something this project is trying to get
        # right in its own numbers, so those keep the original flat-VOLL
        # fallback untouched.
        if voll_true.any() and elec_is_nl[iAk]:
            iI_here = np.where(xc_to_idx == iAk)[0]
            iS_here = np.where(shed_per_elec[:, iAk])[0]
            if iI_here.size > 0 or iS_here.size > 0:
                relief_capacity_parts = []
                relief_cost_parts = []

                if iI_here.size > 0:
                    # -xc_supply is the link's own NOMINAL rating (what it
                    # could carry if its sending node had unlimited surplus) -
                    # on its own that overstates real relief whenever the
                    # sender doesn't actually have that much spare generation
                    # that hour. Bound it by the sender's own real surplus
                    # (net_available_hourly, already known for every node
                    # after pass 1), converted through the link's efficiency
                    # to this node's units - the exact same bound
                    # disp_power_interconnectors.py's own from_surplus already
                    # applies to the REAL flow, so the price this resolves to
                    # and the flow that actually gets dispatched afterward
                    # agree on how much relief is really available. Without
                    # this, a shortage could get priced as "resolved by
                    # relief" from the link's rating alone while the real,
                    # surplus-bounded flow later delivers far less - leaving
                    # demand genuinely unserved without ever pricing at VOLL.
                    nominal_relief = -xc_supply[:, iI_here]
                    from_surplus = np.maximum(net_available_hourly[:, xc_from_elecidx[iI_here]], 0.0)
                    from_surplus_as_relief = from_surplus * xc_efficiencies[iI_here, 0][None, :]
                    relief_capacity_parts.append(np.minimum(nominal_relief, from_surplus_as_relief))
                    from_prices = prices_hourly[:, xc_from_actidx[iI_here]]  # (nH, nI)
                    # Same delivered-cost convention disp_power_interconnectors.py's
                    # own spread formula already uses (from_price/efficiency - vom).
                    relief_cost_parts.append(from_prices / xc_efficiencies[iI_here, 0][None, :] - xc_vom[iI_here][None, :])

                if iS_here.size > 0:
                    # A shedding technology's own curtailable band this hour -
                    # how far its demand can drop from shed_maxdemand_hourly
                    # toward shed_mindemand_hourly - priced at its own VOM,
                    # the same role xc_vom plays for interconnectors above.
                    relief_capacity_parts.append(
                        shed_maxdemand_hourly[:, iS_here] - shed_mindemand_hourly[:, iS_here])
                    relief_cost_parts.append(np.broadcast_to(shed_vom[iS_here][None, :], (nH, iS_here.size)))

                relief_capacity = np.concatenate(relief_capacity_parts, axis=1)  # (nH, nR)
                relief_cost = np.concatenate(relief_cost_parts, axis=1)  # (nH, nR)

                relief_order = np.argsort(relief_cost, axis=1, kind='stable')
                relief_cum = np.cumsum(np.take_along_axis(relief_capacity, relief_order, axis=1), axis=1)
                gap = demand - MOC_volume_at_last
                relief_meets = relief_cum >= gap[:, None]
                relief_any_meets = relief_meets.any(axis=1)
                relief_last = np.where(relief_any_meets, relief_meets.argmax(axis=1), relief_cost.shape[1] - 1)
                relief_marginal_cost = np.take_along_axis(relief_cost, relief_order, axis=1)[hours, relief_last]

                # VOLL is the accepted ceiling price for unmet demand - only
                # prefer relief over it when relief is actually cheaper.
                # Without this, an expensive relief option (its own delivered/
                # curtailment cost above voll) could still get selected purely
                # because it physically closes the gap, quietly reporting a
                # price *above* the ceiling the model is supposed to cap at.
                resolved = voll_true & relief_any_meets & (relief_marginal_cost <= voll)

                if resolved.any():
                    marginal_cost = np.where(resolved, relief_marginal_cost, marginal_cost)
                    voll_true = voll_true & ~resolved

        elec_prices_hourly[:, iAk] = np.where(voll_true, voll, marginal_cost)

    return gen_use_hourly, elec_prices_hourly, net_available_hourly
