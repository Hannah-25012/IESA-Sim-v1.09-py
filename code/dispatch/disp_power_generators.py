# File to dispatch generators
import numpy as np

def disp_power_generators(gen_vom, gen_balance_hourly, gen_availability_hourly,
                          gen_xc_costs_hourly, gen_per_elec, elec_demand_hourly,
                          prices_hourly, voll,
                          xc_to_idx, xc_from_actidx, xc_supply, xc_efficiencies, xc_vom, elec_is_nl):

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
    hours = np.arange(nH)

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

        iG_marginal = MOC_order[hours, MOC_last_gen]

        # Save dispatch parameters
        marginal_cost = gen_cost[hours, iG_marginal]

        # Before falling back to VOLL: a node whose own generator fleet can't
        # meet demand may still have a connected interconnector/transformer
        # able to close the gap - this only ever matters in an hour that's
        # already voll_true, so it can never make a normally-clearing hour's
        # price worse or change which generator sets the marginal price
        # there. Real generator dispatch (gen_use_hourly below) is untouched
        # either way - in a genuine shortage every available generator still
        # runs at full output regardless of whether the residual gap gets
        # priced at VOLL or at the interconnector's delivered cost; the
        # interconnector's own actual flow is decided afterwards by
        # disp_power_interconnectors, using whichever price this resolves to.
        #
        # Scoped to the Netherlands' own nodes (elec_is_nl) - every other
        # country node's own fleet is IESA-Sim's auxiliary representation of
        # a neighboring market, not something this project is trying to get
        # right in its own numbers, so those keep the original flat-VOLL
        # fallback untouched.
        if voll_true.any() and elec_is_nl[iAk]:
            iI_here = np.where(xc_to_idx == iAk)[0]
            if iI_here.size > 0:
                relief_capacity = -xc_supply[:, iI_here]  # (nH, nR) positive deliverable qty at this node
                from_prices = prices_hourly[:, xc_from_actidx[iI_here]]  # (nH, nR)
                # Same delivered-cost convention disp_power_interconnectors.py's
                # own spread formula already uses (from_price/efficiency - vom).
                relief_cost = from_prices / xc_efficiencies[iI_here, 0][None, :] - xc_vom[iI_here][None, :]

                relief_order = np.argsort(relief_cost, axis=1, kind='stable')
                relief_cum = np.cumsum(np.take_along_axis(relief_capacity, relief_order, axis=1), axis=1)
                gap = demand - MOC_volume_at_last
                relief_meets = relief_cum >= gap[:, None]
                relief_any_meets = relief_meets.any(axis=1)
                relief_last = np.where(relief_any_meets, relief_meets.argmax(axis=1), relief_cost.shape[1] - 1)
                relief_marginal_cost = np.take_along_axis(relief_cost, relief_order, axis=1)[hours, relief_last]

                # VOLL is the accepted ceiling price for unmet demand - only
                # prefer interconnector relief over it when relief is actually
                # cheaper. Without this, an expensive relief option (its own
                # delivered cost above voll) could still get selected purely
                # because it physically closes the gap, quietly reporting a
                # price *above* the ceiling the model is supposed to cap at.
                resolved = voll_true & relief_any_meets & (relief_marginal_cost <= voll)

                if resolved.any():
                    marginal_cost = np.where(resolved, relief_marginal_cost, marginal_cost)
                    voll_true = voll_true & ~resolved

        elec_prices_hourly[:, iAk] = np.where(voll_true, voll, marginal_cost)

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

    return gen_use_hourly, elec_prices_hourly
