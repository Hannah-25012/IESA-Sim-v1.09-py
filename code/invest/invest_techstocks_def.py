# File to determine definitive investments in technologies for the current period
import numpy as np

def invest_techstocks_def(dimensions, technologies, tech_stock_original, preliminary_investments, report_yes, iP):

    # Extract parameters
    nP = dimensions['nP']
    nTb = dimensions['nTb']
    tech_entities = technologies.balancers.entities
    inv_cost = technologies.balancers.costs.investments[:, iP]
    tech_stock_min = technologies.balancers.stocks.min[:, iP]
    tech_stock_max = technologies.balancers.stocks.max[:, iP]
    decommissionings = technologies.balancers.decommissionings

    # Preallocate the existing tech stocks
    tech_stock_original = tech_stock_original.reshape(-1, 1)
    tech_stock = tech_stock_original + preliminary_investments
    tech_stock_new = np.zeros((nTb,1))  # Preallocate
    approved_investments = np.zeros((nTb,1))  # Preallocate
    forced_investments = np.zeros((nTb,1))  # Preallocate
    forced_decommissionings = np.zeros((nTb,1))  # Preallocate

    # Force all technologies to fall within min and max limits
    for tech in tech_entities:
        iTb = tech.idx

        # Determine allowed stock
        tech_stock_new[iTb,0] = min(max(tech_stock_min[iTb], tech_stock[iTb]), tech_stock_max[iTb])

        # Approve investments that do not violate the constraint
        approved_investments[iTb,0] = min(preliminary_investments[iTb,0],
                                        tech_stock_new[iTb,0] - tech_stock_original[iTb,0])

        # Calculate the delta
        delta_stock = tech_stock_new[iTb,0] - tech_stock[iTb]

        # Check the direction of the delta
        if delta_stock > 0:  # If the delta is positive, invest
            forced_investments[iTb,0] = max(0, delta_stock)
        elif delta_stock < 0:  # If the delta is negative, decommission
            forced_decommissionings[iTb,0] = -min(0, delta_stock)

        # Define the final stocks
        tech_stock[iTb,0] = tech_stock_new[iTb,0]

    # Make all primary energy equal to tech stock max (it does not reflect on investments)
    primary_decommissionings = np.zeros((nTb,1))  # Preallocate
    for tech in tech_entities:
        iTb = tech.idx

        # 'XC Trade' is IESA-Opt's own category for cross-border trade
        # technologies (electricity import/export aggregates) - IESA-Sim's
        # own workbooks never use it as a category (only as a subsector,
        # with a real category of their own: Primary/External/Exports/
        # Conversion - see dbcompare-backend's technologies entry), so this
        # only ever matches an Opt-sourced or merged-database technology,
        # never a native Sim one. Added alongside 'Primary' because a
        # merged database's IESA-Opt-anchored XC Trade technology (e.g.
        # "Electricity Import from EU - Power NL") has no Sim-side
        # equivalent to inherit a 'Primary' categorization from (Sim
        # models the same cross-border capacity as several separate
        # bilateral, per-country technologies instead of Opt's one
        # aggregate) - without this, that technology is left subject to
        # normal cost-competitive investment, which spirals into a
        # circular scarcity-pricing trap (its own "fuel" cost is the price
        # of the electricity it imports, already inflated by its own
        # under-investment) instead of representing existing interconnector
        # capacity the way Opt's own model treats it (IESA-Opt's own
        # solver doesn't re-justify import/export capacity economically
        # either - it links import and export capacity together as the
        # same physical line, see julia-backend/src/model/stock.jl's
        # linked_investments_XC).
        # subsector='Inland generation' is IESA-Opt's own placeholder for
        # externally-sourced generation feeding the shared cross-border
        # pool (e.g. "Inland generation - Power EU") - conceptually the
        # same "already exists, doesn't need investment justification"
        # role as a Primary resource (it represents capacity elsewhere in
        # Europe supplying the pool, not a new plant IESA-Sim's own model
        # would ever build), but IESA-Opt's own source data leaves it with
        # stock_initial=0 and no cost data at all (confirmed directly
        # against IESA-Opt's own database, not a merge gap), so normal
        # investment logic can never bring it online. Like 'XC Trade'
        # above, this subsector value never appears in IESA-Sim's own
        # workbooks (checked directly), so this only ever matches an
        # Opt-sourced or merged-database technology - zero effect on a
        # native Sim run.
        #
        # subsector='Transformer' catches IESA-Opt's own HV<->LV grid
        # transformers (e.g. "Transformer from LV to HV - Power NL") - a
        # dbcompare-backend-only reclassification (see /unify's post-merge
        # fixup) that splits subsector='XC Trade' into genuine cross-border
        # trade (kept as 'XC Trade') and same-country links like these
        # transformers (relabeled 'Transformer'), based on each technology's
        # actual to/from activity nodes rather than trusting the shared Opt
        # subsector text. IESA-Sim's own model has no HV/LV split at all (a
        # single "Electricity NL" node, so this subsector value can never
        # arise from native Sim data either), so like the other two
        # conditions above this only ever matches an Opt-sourced or
        # merged-database technology. These have real investment cost data
        # (unlike Inland generation), but invest_power_technologies.py's own
        # investment gate is built entirely around POWER GENERATOR economics
        # (capture rate, normalized utilization factor, cash-flow-based
        # payback time - see its own condition_1/condition_2) - concepts
        # that don't apply to a transformer, which doesn't generate or sell
        # anything, just moves capacity between two price tiers. Without
        # this, that gate never fires for a transformer (confirmed: capture
        # rate/utilization factor are generator-only metrics, never
        # populated for a transformer, so both conditions read as false),
        # so new_investments for it stays permanently 0 - one merged run
        # showed "Transformer from LV to HV" stuck at 0 GW installed
        # capacity forever and "Transformer from HV to LV" decommissioning
        # from 9 GW to 0 GW with no reinvestment, severing the HV/LV link
        # entirely and pinning the LV electricity price at a scarcity
        # ceiling. Treated the same as existing grid capacity, matching how
        # IESA-Opt's own solver doesn't re-justify import/export capacity
        # economically either (see the 'XC Trade' comment above).
        # subsector='Undispatched' is IESA-Opt's own VOLL (Value of Lost
        # Load) accounting technology - a scarcity-price placeholder, not a
        # real generator anyone invests in. IESA-Sim's own native workbook
        # already gives its own VOLL technologies (one per country) a
        # stock_initial set directly to their intended constant value, so
        # normal investment logic never has anything to do for them
        # (confirmed: technology_investments shows exactly 0 for every
        # native VOLL technology in every period). A merged database's
        # unmatched "Undispatched Electricity (VOLL) - Power NL" instead
        # gets stock_initial correctly zero-filled (see dbcompare-backend's
        # own zero-fill fix) but keeps a real, nonzero technology_stocks.max
        # from period 1 - so normal investment logic sees a gap from 0 to
        # that max and "invests" into it once, charging a one-time capital
        # cost for a technology that isn't a physical asset at all (one
        # merged run: 30 units x 2,389 MEUR/unit = 71.7 BEUR, on its own
        # roughly 43% of that run's entire period-1 capital cost line).
        # Guaranteeing it the same way as native Sim's own VOLL technologies
        # already effectively behave removes that phantom charge.
        if ('Primary' in tech.category or 'XC Trade' in tech.category or
                tech.subsector == 'Inland generation' or tech.subsector == 'Transformer' or
                tech.subsector == 'Undispatched'):
            tech_stock[iTb] = tech_stock_max[iTb]
            primary_decommissionings[iTb] = tech_stock_max[iTb]

            # The comment above ("it does not reflect on investments") isn't
            # actually true for approved_investments/forced_investments -
            # both were already computed by the first loop above, from
            # whatever delta happened between tech_stock_original and the
            # min/max-clamped tech_stock, before this loop even runs, and
            # get saved into technologies.balancers.investments regardless
            # of what this loop does to tech_stock afterwards. Scoped to
            # 'Undispatched' only (not the Primary/XC Trade/Inland
            # generation/Transformer conditions above, which predate this
            # fix and whose own investment bookkeeping also feeds
            # invest_decommissioning.py and results_policy_cashflows.py -
            # zeroing it there measurably changes native Sim's own results
            # for real Primary technologies like "Imported Crude Oil" from
            # 2030 onward, a materially bigger change than this fix is
            # meant to make). VOLL's own investment bookkeeping is already
            # exactly 0 on native Sim (its stock_initial is already its
            # full constant value there), so zeroing it only ever matters
            # for a merged database's unmatched VOLL technology, whose
            # stock_initial is correctly zero-filled but keeps a real,
            # nonzero technology_stocks.max from period 1 - normal
            # investment logic sees a gap from 0 to that max and "invests"
            # into it once, charging a one-time capital cost for a
            # technology that isn't a physical asset at all (one merged
            # run: 30 units x 2,389 MEUR/unit = 71.7 BEUR).
            if tech.subsector == 'Undispatched':
                approved_investments[iTb, 0] = 0.0
                forced_investments[iTb, 0] = 0.0

        elif 'Emission' in tech.category:
            # "Indirect - <gas> <sector>" (e.g. "Indirect - CH4 Agriculture")
            # is IESA-Opt's own baseline non-energy-GHG accounting - tracking
            # what gets emitted before any abatement choice, not a mitigation
            # technology itself (contrast "MACC Component ..." nearby, which
            # genuinely is an abatement decision with real investment
            # economics). These technologies don't exist in IESA-Sim's own
            # workbook at all (checked directly - zero native rows for this
            # name pattern), so this only ever matches an Opt-sourced or
            # merged-database technology. Every one of them has a nonzero
            # investment cost in Opt's own data despite being pure
            # accounting, so the inv_cost==0 branch below never catches
            # them, and the same zero-stock_initial-but-nonzero-max gap as
            # 'Undispatched' above charges a phantom capital cost for
            # baseline emissions tracking that was never a capacity decision.
            if inv_cost[iTb] == 0 or tech.name.startswith('Indirect - '):
                tech_stock[iTb] = tech_stock_max[iTb]
                primary_decommissionings[iTb] = tech_stock_max[iTb]
                # Same first-loop bookkeeping gap as the guaranteed-
                # availability branch above - see its own comment.
                approved_investments[iTb, 0] = 0.0
                forced_investments[iTb, 0] = 0.0

            if 'Emission' in tech.sector:
                tech_stock[iTb] = 5000
                primary_decommissionings[iTb] = 5000

    # Update primary decommissioning for next period
    if iP + 1 < nP:
        decommissionings[:, iP + 1] += primary_decommissionings[:, 0]

    # Report the definitive stocks if requested
    if report_yes:
        print(f"{'Technology':60s}, {'Tech Stock':10s}")
        for tech in tech_entities:
            print(f"{tech.id:60s}, {tech_stock[tech.idx]:10.2f}")

    # Save variables
    technologies.balancers.stocks.evolution[:, iP] = tech_stock.flatten()
    technologies.balancers.investments[:, iP] = approved_investments.flatten() + forced_investments.flatten()
    technologies.balancers.decommissionings = decommissionings

    return technologies, forced_decommissionings
