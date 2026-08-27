# Determine the investment potential per technology
# THINK: the logic of the code seems to prioritize deployment first, then retrofitting. Also, I don't exactly understand what happens with room_to_invest if retrofit option is chosen
import numpy as np

def invest_investment_potential(dimensions, technologies, tech_stock_exist, retrofit_potential, ip):

    # Extract parameters
    nTb = dimensions['nTb']
    tech_entities = technologies.balancers.entities
    tech_stock_max = technologies.balancers.stocks.max[:, ip]

    # Obtain investment potentials
    investment_potential = np.zeros(nTb)  # Preallocate
    for tech in tech_entities:

        # Obtain the raw room to invest
        room_to_invest = tech_stock_max[tech.idx] - tech_stock_exist[tech.idx]

        # Check if there is a deploy limitation or a retrofit option
        if tech.stock_deploy > 0:
            investment_potential[tech.idx] = min(tech.stock_deploy, room_to_invest)
        elif retrofit_potential[tech.idx] > 0:
            investment_potential[tech.idx] = retrofit_potential[tech.idx]
        else:
            # KNOWN LIMITATION (documented, not fixed): with no stock_deploy
            # and no retrofit option, a technology's investment potential is
            # its entire remaining headroom to tech_stock_max, with no
            # per-period build-rate limit at all. This is NOT a blanket
            # merged-vs-native split - power generation technologies (e.g.
            # CCGT, Onshore/Offshore Wind, Solar PV) correctly carry a real,
            # non-zero stock_deploy through the merge wherever the curated
            # technology-alias system matches them to their (differently-
            # numbered) native Sim counterpart - confirmed via the merged
            # technologies table's own sim_id column (e.g. merged "CCGT -
            # Power NL" -> sim_id "PNL01_01", stock_deploy 10 GW/period,
            # inherited correctly). The actual gap is narrower and specific:
            # IESA-Sim's own native workbook itself has stock_deploy=0 for its
            # hydrogen production technologies (Gas Reforming, Alkaline/PEM
            # Electrolyzer, ...) - a genuine gap in the source data, present
            # identically on the merged side too (these are 1:1 self-matched,
            # sim_id==opt_id, no aliasing involved). Native Sim never exposes
            # this because its own demand trajectory never creates a hydrogen
            # investment gap large enough to need a lump-sum fill; a merged
            # run's demand dynamics can (see
            # dbcompare-backend/MERGED_DB_BACKLOG.md) - confirmed real-world
            # effect: the Hydrogen HD scarcity fallback
            # (invest_energy_technologies.py, "Facilitate the resources...")
            # forced Alkaline Electrolyzer - Hydrogen Production from 0
            # straight to ~1774 units of use in a single period once fixed to
            # pick a real producer instead of an export technology - which,
            # via that electrolyzer's own -1.57 electricity-per-unit-hydrogen
            # energy-balance coefficient, created a ~2786-unit electricity
            # demand spike in one period that even the power sector's own
            # correctly-paced investment logic couldn't absorb, flooding
            # VOLL. Not a bug in this function or the fallback - both do
            # exactly what they're told - but a genuinely missing constraint
            # in the source data for this specific technology family. Left
            # undecided deliberately: whether to backfill hydrogen
            # technologies' stock_deploy with an estimated ramp rate (a data
            # decision) or add a new default pacing rule here for
            # technologies missing it (a modeling decision) is an open
            # choice, not a correction of something already wrong.
            investment_potential[tech.idx] = room_to_invest

    return investment_potential
