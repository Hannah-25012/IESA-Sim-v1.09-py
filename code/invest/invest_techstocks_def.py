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
        if ('Primary' in tech.category or 'XC Trade' in tech.category or
                tech.subsector == 'Inland generation' or tech.subsector == 'Transformer'):
            tech_stock[iTb] = tech_stock_max[iTb]
            primary_decommissionings[iTb] = tech_stock_max[iTb]

        elif 'Emission' in tech.category:
            if inv_cost[iTb] == 0:
                tech_stock[iTb] = tech_stock_max[iTb]
                primary_decommissionings[iTb] = tech_stock_max[iTb]

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
