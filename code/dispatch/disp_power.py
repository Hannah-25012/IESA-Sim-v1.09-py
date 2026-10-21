# File to determine the power dispatch
import os
import numpy as np
from disp_initialize_power import disp_initialize_power
from disp_power_generators import disp_power_generators
from disp_power_shedding import disp_power_shedding
from disp_power_loadshifting import disp_power_loadshifting
from disp_power_batteries import disp_power_batteries
from disp_power_interconnectors import disp_power_interconnectors

# Blend factor between the previous and the newly computed hourly demand each
# iteration of the power-dispatch loop: 1.0 = jump straight to the new demand
# (original behaviour), smaller = damped. Price-responsive demand (shedding,
# load-shifting, batteries) reacts to the price the generators just produced,
# which depends on the demand they were given, so a full jump can cycle
# between two states every iteration instead of settling. Override with the
# IESA_DEMAND_RELAXATION environment variable for experiments.
DEMAND_RELAXATION = float(os.environ.get("IESA_DEMAND_RELAXATION", "1.0"))

def disp_power(dimensions, parameters, activities, technologies, profiles, tech_use_hourly, prices_hourly, iP, nId):

    # Extract parameters
    nDy = dimensions['nDy']
    nHd = dimensions['nHd']
    nH = dimensions['nH']
    nIp = dimensions['nIp']
    nAk = dimensions['nAk']
    voll = parameters.voll
    min_spread = parameters.min_spread
    activities_label = activities.labels
    activities_elec = activities.electricity.names
    # VOLL-avoidance via interconnector and shedding relief (see
    # disp_power_generators.py) is scoped to the Netherlands' own electricity
    # nodes only - the model's actual focus. Every other country node
    # (Electricity DE/BE/DK/GB/NO) and the generic EU pool are IESA-Sim's own
    # auxiliary representations of neighboring markets, not something this
    # project is trying to get right in their own numbers; leaving them on
    # the untouched flat-VOLL fallback keeps their reported prices/
    # investments/emissions exactly as before.
    elec_is_nl = np.array([name.startswith('Electricity NL') for name in activities_elec])

    # Initialize power dispatch
    if nId:
        print('---Initializing power dispatch...')
    (
        gen_vom, gen_balance_hourly, gen_availability_hourly,
        gen_xc_costs_hourly, gen_per_elec, elec_demand_hourly,
        shed_guarantee, shed_maxvolume_hourly, shed_minvolume_hourly, shed_per_elec,
        shed_maxdemand_hourly, shed_mindemand_hourly, shed_multiplier, shed_vom,
        loadshifts_efficiencies, loadshifts_capacities, loadshifts_min, loadshifts_per_uoa,
        loadshifts_range, loadshifts_per_elec, loadshifts_demand_hourly,
        bat_efficiency, bat_capacity, bat_volume, bat_per_elec, bat_vom, bat_stock,
        xc_efficiencies, xc_vom, xc_per_elec, xc_demand, xc_supply, elec_cogenerated,
        xc_to_idx, xc_from_actidx, xc_from_elecidx, gen_is_voll,
        tech_generators_coord, tech_shedding_coord, tech_loadshifts_coord,
        tech_batteries_coord, tech_interconnectors_coord
    ) = disp_initialize_power(dimensions, activities, technologies, profiles, tech_use_hourly, iP)

    # Pinned annual flows for interconnectors (use_min == use_max in the
    # database, as a PJ total for this period) -> constant fraction of each
    # link's nominal annual capacity (stock x cap2act); NaN = free dispatch.
    xc_fixed_fraction = np.full(int(np.sum(tech_interconnectors_coord)), np.nan)
    _stocks = technologies.balancers.stocks
    if hasattr(_stocks, 'use_min') and xc_fixed_fraction.size:
        _idx = np.where(tech_interconnectors_coord)[0]
        _umin, _umax = _stocks.use_min[_idx, iP], _stocks.use_max[_idx, iP]
        _nominal = (technologies.balancers.stocks.evolution[_idx, iP] * technologies.balancers.cap2acts[_idx])
        _pinned = np.isfinite(_umin) & np.isfinite(_umax) & (_umin == _umax) & (_nominal > 0)
        xc_fixed_fraction[_pinned] = np.clip(_umin[_pinned] / _nominal[_pinned], 0.0, 1.0)

    # Display header of the price table
    if nId:
        print('---Beginning of the power dispatch iterations...')
        print('==================== Average electricity price per node in EUR/MWh ====================')
        print('======================= Average electricity load per node in TWh ======================')
        print(f"{'Iter':6}, {'Type':10}", end='')
        for iAk in range(nAk):
            print(f"{activities_elec[iAk]:24}", end='')
        print()

    # Begin the iterative loop
    elec_demand_hourly_new = elec_demand_hourly.copy()
    bat_demand_elec_hourly = np.zeros((nH, nAk))  # batteries' previous-iteration demand (none yet)
    # Per-node demand excluding interconnector/transformer terms (shedding,
    # load-shifting and battery terms included) - the baseline a link's
    # real gap/spare is measured against, see disp_power_generators.py.
    elec_demand_ex_xc_hourly = elec_demand_hourly.copy()
    for iIp in range(nIp):

        # Dispatch generators
        gen_use_hourly, elec_prices_hourly, net_available_hourly = disp_power_generators(  # = spare vs demand without link flows
            gen_vom, gen_balance_hourly, gen_availability_hourly,
            gen_xc_costs_hourly, gen_per_elec, elec_demand_hourly_new,
            prices_hourly, voll,
            xc_to_idx, xc_from_actidx, xc_from_elecidx, xc_supply, xc_efficiencies, xc_vom, elec_is_nl,
            shed_per_elec, shed_maxdemand_hourly, shed_mindemand_hourly, shed_vom, gen_is_voll,
            elec_demand_ex_xc_hourly
        )

        # Display iteration info for the last balancing loop
        if nId:
            print(f"{iIp:6d} {'Price':10}", end='')
            for iAk in range(nAk):
                print(f"{np.mean(elec_prices_hourly[:, iAk]) * 3.6:24.2f}", end='')
            print()
            print(f"{iIp:6d} {'Load':10}", end='')
            for iAk in range(nAk):
                print(f"{(sum(elec_demand_hourly_new[:, iAk]) + elec_cogenerated[iAk, 0]) / 3.6:24.2f}", end='')
            print()

        # Dispatch shedding technologies - ONCE, in the first iteration, then
        # frozen. Re-deciding the schedule from every iteration's prices makes
        # it chase prices that move because of that very schedule (all
        # flexible consumers pile onto the cheapest hours, those hours turn
        # into shortages and price spikes, the schedule flips away, prices
        # collapse, it flips back - a period-2 cycle the iteration count only
        # freezes at a random phase). The schedule is built from the first
        # iteration's prices and each node's real spare generation after all
        # minimum demands are reserved, so it never creates a shortage that
        # was not already there - the simulation counterpart of IESA-Opt's
        # shedding being a bounded term inside the hourly balance.
        if sum(tech_shedding_coord) and iIp == 0:
            spare_for_shedding = np.array(net_available_hourly, dtype=float, copy=True)
            for iAk in range(nAk):
                on_node = shed_per_elec[:, iAk]
                if on_node.any():
                    spare_for_shedding[:, iAk] -= shed_mindemand_hourly[:, on_node].sum(axis=1)
            shed_use_hourly = disp_power_shedding(
                shed_guarantee, shed_maxvolume_hourly,
                shed_minvolume_hourly, shed_per_elec, elec_prices_hourly,
                shed_maxdemand_hourly, shed_mindemand_hourly, spare_for_shedding
            )

        # Dispatch load-shifting technologies
        if sum(tech_loadshifts_coord) > 0:
            loadshifts_demand_hourly = disp_power_loadshifting(
                loadshifts_efficiencies, loadshifts_capacities,
                loadshifts_min, loadshifts_range, loadshifts_per_elec,
                loadshifts_demand_hourly, elec_prices_hourly, nDy, nHd
            )

        # Dispatch batteries
        if sum(tech_batteries_coord) > 0:
            # Charging only where the node has real surplus (Opt's hourly
            # balance never lets storage charge into a shortage). The spare
            # is measured WITHOUT the batteries' own previous demand - the
            # surplus they would charge from - so it does not move when the
            # batteries do. Every battery on a node reserves the node's full
            # combined charging power together, so they cannot all claim
            # the same spare hour.
            safe_bat_eff = np.where(np.asarray(bat_efficiency, dtype=float) != 0, bat_efficiency, 1.0)
            bat_charge_power = np.asarray(bat_stock, dtype=float) * np.asarray(bat_capacity, dtype=float) / safe_bat_eff
            bat_node = np.array([np.where(bat_per_elec[iB, :])[0][0] for iB in range(len(bat_stock))])
            node_charge_power = np.zeros(nAk)
            np.add.at(node_charge_power, bat_node, bat_charge_power)
            spare_ex_battery = net_available_hourly + bat_demand_elec_hourly
            bat_charge_allowed = spare_ex_battery[:, bat_node] >= node_charge_power[bat_node][None, :]
            bat_use_new, bat_demand_new = disp_power_batteries(
                bat_efficiency, bat_capacity, bat_volume,
                bat_vom, bat_stock, bat_per_elec, elec_prices_hourly, min_spread,
                bat_charge_allowed
            )
            # Running average of the batteries' schedule over the passes
            # (weight 1/(pass+1): the first pass is taken whole, each later
            # one counts a little less). A battery plans against the previous
            # pass's prices, and its own schedule is what moved those prices,
            # so taking each best response in full makes the whole fleet flip
            # between two opposite schedules every pass (2030, when storage
            # jumps from ~0.3 to ~25 GW in one period: demand change per pass
            # ~2x the total battery demand). Averaging the schedule is the
            # standard remedy for price-taking best-response loops and lets
            # the price/schedule pair settle instead of flipping.
            if iIp == 0 or 'bat_use_hourly' not in locals():
                bat_use_hourly, bat_demand_elec_hourly = bat_use_new, bat_demand_new
            else:
                w_avg = 1.0 / (iIp + 1)
                bat_use_hourly = bat_use_hourly + w_avg * (bat_use_new - bat_use_hourly)
                bat_demand_elec_hourly = bat_demand_elec_hourly + w_avg * (bat_demand_new - bat_demand_elec_hourly)

        # Dispatch interconnectors
        xc_use_hourly = disp_power_interconnectors(
            xc_efficiencies, xc_vom, xc_per_elec, elec_prices_hourly, xc_demand, net_available_hourly,
            xc_fixed_fraction
        )

        # Adjust demand for each electricity activity
        conv_change = 0.0
        conv_total = 0.0
        conv_ex = 0.0
        conv_xc = 0.0
        for iAk in range(nAk):
            prev_demand_new = elec_demand_hourly_new[:, iAk].copy()
            prev_demand_ex_xc = elec_demand_ex_xc_hourly[:, iAk].copy()

            # Identify shedding technologies and interconnectors in the node
            is_shedding = shed_per_elec[:, iAk]
            is_loadshifts = loadshifts_per_elec[:, iAk]
            is_interconnect_from = np.sum(xc_per_elec[:, iAk, :], axis=0).T[:, None].astype(bool)
            is_interconnect_to = np.sum(xc_per_elec[iAk, :, :], axis=0).T[:, None].astype(bool)

            # Add base demand
            elec_demand_hourly_new[:, iAk] = elec_demand_hourly[:, iAk]

            # Add demand of all shedding technologies
            if np.sum(tech_shedding_coord) > 0:  # Check if there are any active shedding technologies
                elec_demand_hourly_new[:, iAk] += np.sum(
                    shed_mindemand_hourly[:, is_shedding] +
                    shed_use_hourly[:, is_shedding] * (
                        shed_maxdemand_hourly[:, is_shedding] - shed_mindemand_hourly[:, is_shedding]
                    ),
                    axis=1  # Sum along rows (equivalent to MATLAB's sum(..., 2))
                )

            # Add shifting technology demand
            if sum(tech_loadshifts_coord) > 0:
                elec_demand_hourly_new[:, iAk] += loadshifts_demand_hourly[:, is_loadshifts].sum(axis=1)

            # Add battery demand
            if sum(tech_batteries_coord) > 0:
                elec_demand_hourly_new[:, iAk] += bat_demand_elec_hourly[:, iAk]

            elec_demand_ex_xc_hourly[:, iAk] = elec_demand_hourly_new[:, iAk]

            # Add interconnector demand and supply
            elec_demand_hourly_new[:, iAk] += (
                xc_demand[:, is_interconnect_from.ravel()] * xc_use_hourly[:, is_interconnect_from.ravel()]
            ).sum(axis=1) + (
                xc_supply[:, is_interconnect_to.ravel()] * xc_use_hourly[:, is_interconnect_to.ravel()]
            ).sum(axis=1)

            conv_change += np.abs(elec_demand_hourly_new[:, iAk] - prev_demand_new).sum()
            conv_total += np.abs(elec_demand_hourly_new[:, iAk]).sum()
            conv_ex += np.abs(elec_demand_ex_xc_hourly[:, iAk] - prev_demand_ex_xc).sum()
            conv_xc += np.abs((elec_demand_hourly_new[:, iAk] - elec_demand_ex_xc_hourly[:, iAk]) - (prev_demand_new - prev_demand_ex_xc)).sum()
            if DEMAND_RELAXATION != 1.0:
                elec_demand_hourly_new[:, iAk] = prev_demand_new + DEMAND_RELAXATION * (elec_demand_hourly_new[:, iAk] - prev_demand_new)
                elec_demand_ex_xc_hourly[:, iAk] = prev_demand_ex_xc + DEMAND_RELAXATION * (elec_demand_ex_xc_hourly[:, iAk] - prev_demand_ex_xc)

        # Relative demand change this iteration (0 = settled) - one line per
        # iteration so the number of passes needed to converge can be read
        # straight off the log.
        if nId:
            print(f"CONV iP={iP} iter={iIp} relax={DEMAND_RELAXATION} resid={conv_change / max(conv_total, 1e-12):.5f} ex_xc={conv_ex / max(conv_total, 1e-12):.5f} links={conv_xc / max(conv_total, 1e-12):.5f}", flush=True)

    # Display message
    if nId:
        print('---End of the power dispatch iterations.')

    # Save the variables
    # prices_hourly
    activities_elec_coord = [label == 'Electricity' for label in activities_label]
    prices_hourly[:, activities_elec_coord] = elec_prices_hourly

    # tech_use_hourly
    tech_use_hourly[:, tech_generators_coord] = gen_use_hourly

    if sum(tech_shedding_coord):
        # A shedding technology with zero stock this period (see
        # disp_initialize_power's own ref_profile guard) has shed_multiplier
        # == 0 too - dividing by it here would reintroduce the same NaN this
        # is meant to avoid. That technology's own shed_*demand_hourly are
        # already all zero in that case, so 0/0 -> 0 is the correct result,
        # not inf/NaN.
        shed_multiplier_row = np.ones((nH, 1)) @ shed_multiplier.reshape(1, -1)
        numerator = shed_mindemand_hourly + shed_use_hourly * (shed_maxdemand_hourly - shed_mindemand_hourly)
        tech_use_hourly[:, tech_shedding_coord] = np.divide(
            numerator, shed_multiplier_row,
            out=np.zeros_like(numerator), where=shed_multiplier_row != 0,
        )

    if sum(tech_loadshifts_coord) > 0:
        tech_use_hourly[:, tech_loadshifts_coord] = (
            loadshifts_demand_hourly / (
                np.ones((nH, 1)) @ loadshifts_per_uoa.reshape(1, -1) + np.finfo(float).eps
            )
        )

    if sum(tech_batteries_coord) > 0:
        tech_use_hourly[:, tech_batteries_coord] = bat_use_hourly

    tech_use_hourly[:, tech_interconnectors_coord] = xc_demand * xc_use_hourly

    return tech_use_hourly, prices_hourly
