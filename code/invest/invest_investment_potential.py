# Determine the investment potential per technology
# THINK: the logic of the code seems to prioritize deployment first, then retrofitting. Also, I don't exactly understand what happens with room_to_invest if retrofit option is chosen
import numpy as np

# Default per-period build-rate cap for a technology with no stock_deploy set
# and no retrofit option - see the else-branch below. Chosen as a moderate,
# round rule-of-thumb (a technology can go from zero to its full stock_max
# over a minimum of 5 periods = 25 years at this pace), not a researched
# figure the way the Netherlands hydrogen-electrolyzer stock_deploy values
# were - tune this if a specific technology family needs a tighter/looser
# default. Applied consistently in invest_energy_technologies.py's fallback
# tiers too (DEFAULT_STOCK_DEPLOY_FRACTION import) so the same technology
# can't be paced here and then bypass that pacing there.
DEFAULT_STOCK_DEPLOY_FRACTION = 0.2

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
            # FORMERLY a known, documented limitation: with no stock_deploy
            # and no retrofit option, investment potential used to be the
            # technology's entire remaining headroom to tech_stock_max, with
            # no per-period build-rate limit at all - confirmed to let the
            # scarcity fallback (invest_energy_technologies.py) force a
            # technology from 0 to its full ceiling in one period (Gas
            # Reforming, Solid Oxide Electrolyzer - Hydrogen Production, each
            # in turn, once the previous one got a real stock_deploy value
            # backfilled and the fallback simply moved to the next unpaced
            # candidate). Rather than keep chasing individual technologies as
            # each one gets exposed, fall back to a default per-period cap
            # (DEFAULT_STOCK_DEPLOY_FRACTION of tech_stock_max) instead of
            # unlimited headroom - covers every technology missing
            # stock_deploy at once, not just the ones already found.
            investment_potential[tech.idx] = min(
                DEFAULT_STOCK_DEPLOY_FRACTION * tech_stock_max[tech.idx], room_to_invest
            )

    return investment_potential
