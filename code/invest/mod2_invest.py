# Module to determine investments in technologies
import numpy as np
from invest_retrofit_potential import invest_retrofit_potential
from invest_investment_potential import invest_investment_potential
from invest_drivers_sufficiency import invest_drivers_sufficiency
from invest_tech_LCOPs import invest_tech_LCOPs
from invest_multicriteria_performance import invest_multicriteria_performance
from invest_tech_choices_per_act import invest_tech_choices_per_act
from invest_driver_technologies import invest_driver_technologies
from invest_techstock_check import invest_techstock_check
from invest_nerghg_technologies import invest_nerghg_technologies
from invest_ccus_technologies import invest_ccus_technologies
from invest_energy_technologies import invest_energy_technologies
from invest_power_technologies import invest_power_technologies
from invest_techstocks_def import invest_techstocks_def
from invest_decommissioning import invest_decommissioning

def mod2_invest(dimensions, parameters, activities, technologies, agents, policies, iP, foresight_reference=None,
                 frozen_mix=False, leg_mode=False, debug_dump=None):

    # Extract parameters
    if iP > 0:
        tech_stock_exist = technologies.balancers.stocks.evolution[:, iP - 1]
    else:
        tech_stock_exist = technologies.balancers.stocks.initial
    decommissionings = technologies.balancers.decommissionings[:, iP]

    # === Decommission the technologies in the period ===
    print("--Updating the existing stocks after each period decommissioning...")
    tech_stock_new_original = tech_stock_exist - decommissionings
    tech_stock_new = tech_stock_new_original.copy()

    # === Determine the retrofitting potentials ===
    print("--Determining retrofitting potentials...")
    retrofit_sources, retrofit_options, retrofit_potential, retrofit_cost = invest_retrofit_potential(
        dimensions, technologies, tech_stock_new
    )

    # === Determine investment potentials ===
    print("--Determining investment potentials...")
    investment_potential = invest_investment_potential(
        dimensions, technologies, tech_stock_new, retrofit_potential, iP
    )

    # === Determine the driver gaps and invest ===
    print("--Determining the gaps to meet driver activities volumes...")
    activity_gap = invest_drivers_sufficiency(
        dimensions, activities, technologies, tech_stock_new, False, iP
    )

    # === Obtain the technology LCOPs of the period ===
    print("--Obtaining LCOPs of technologies...")
    technologies = invest_tech_LCOPs(
        dimensions, activities, technologies, policies, retrofit_potential, retrofit_cost, iP,
        foresight_reference=foresight_reference
    )

    # === Call the multi criteria performance function ===
    print("--Measuring multi criteria performance...")
    technologies = invest_multicriteria_performance(
        dimensions, activities, technologies, agents, iP,
        foresight_reference=foresight_reference
    )

    # === Call the technology choices function ===
    print("--Determining technology choices per agent populations...")
    # tech_stock_new here is this period's stock net of its own planned
    # decommissioning but before any new investment - the right basis for
    # frozen_mix's "current fleet shares", matching what invest_drivers/
    # energy_sufficiency above used to size the gap being filled.
    technologies, tech_choices = invest_tech_choices_per_act(
        dimensions, activities, technologies, agents, iP,
        tech_stock=tech_stock_new, frozen_mix=frozen_mix
    )

    # === Determine the new stocks, the old stocks ===
    print("--Determining the required investments to meet driver activities volumes...")
    tech_stock_new, investments_drivers = invest_driver_technologies(
        dimensions, activities, technologies, tech_stock_new, investment_potential, activity_gap, tech_choices, iP,
        frozen_mix=frozen_mix
    )

    # === Check if there are negative investments ===
    if (investments_drivers < 0).any():
        print("!!!Negative investments within drivers routine!!!")
        input("Press Enter to continue...")

    # === Determine the sufficiency of the new driver stocks ===
    print("--Revising that tech stocks are respecting constraints for driver activities...")
    tech_stock_new = invest_techstock_check(
        dimensions, activities, technologies, tech_stock_new, tech_stock_exist, False, iP
    )

    # === Invest in nonER-GHG emission technologies ===
    print("--Determining the required investments to mitigate non-energy related GHG...")
    tech_stock_new, investments_nerghg = invest_nerghg_technologies(
        dimensions, activities, technologies, tech_stock_new, investment_potential, tech_choices, iP,
        frozen_mix=frozen_mix
    )

    # === Invest in CCUS technologies ===
    print("--Determining the required investments to mitigate non-energy related GHG...")
    tech_stock_new, investments_ccus = invest_ccus_technologies(
        dimensions, activities, technologies, tech_stock_new, investment_potential, tech_choices, iP
    )

    # === Determine the energy investments ===
    print("--Determining the required investments to satisfy energy activities...")
    technologies, activities, tech_stock_new, investments_energy = invest_energy_technologies(
        dimensions, activities, technologies, tech_stock_new, investment_potential, tech_choices, iP,
        frozen_mix=frozen_mix
    )

    # Check if there are negative investments
    if (investments_energy < 0).any():
        print("!!!Negative investments within energy routine!!!")
        input("Press Enter to continue...")

    # === Invest in power activities (but not for the first period) ===
    # leg_mode: the first period of a leg that continues a finished run is a normal period (leg_carry.py supplies the previous period's
    # generator indicators), so power investment runs there too; only the true first period of the model skips it.
    if iP > 0 or leg_mode:
        print("--Investing in power generation technologies...")
        investments_power = invest_power_technologies(
            dimensions, parameters, activities, technologies, agents, tech_stock_new, iP,
            frozen_mix=frozen_mix
        )
    else:
        investments_power = np.zeros_like(investments_energy)

    # Check if there are negative investments
    if (investments_power < 0).any():
        print("!!!Negative investments within power routine!!!")
        input("Press Enter to continue...")

    # === Preliminary investments ===
    print("--Defining final investments...")
    preliminary_investments = (
        investments_drivers + investments_nerghg + investments_ccus + investments_energy + investments_power
    )

    # === Determine the definitive tech stocks that will be used for the operation of the system ===
    print("--Determining the definitive tech Stocks...")
    technologies, forced_decommissionings = invest_techstocks_def(
        dimensions, technologies, tech_stock_new_original, preliminary_investments, False, iP
    )

    # === Calculate the decommissionings due to investments ===
    print("--Determining the decommissionings based on the investments...")
    # Pass tech_stock_new_original (net of this period's already-planned
    # decommissioning), not the raw tech_stock_exist: retrofit_potential above
    # was computed against that same net figure (via tech_stock_new), so
    # checking a retrofit source's available stock against the raw,
    # pre-decommissioning figure here would let this period's planned
    # decommissioning and retrofit-sourced decommissioning both draw against
    # the same units and jointly overdraw the source technology's stock.
    technologies = invest_decommissioning(
        dimensions, activities, technologies, forced_decommissionings, retrofit_sources,
        retrofit_options, retrofit_potential, tech_stock_new_original, iP
    )

    if debug_dump:
        _dump_invest_state(debug_dump, activities, technologies, iP, dict(
            tech_stock_exist=tech_stock_exist, decommissionings=decommissionings, tech_stock_new_original=tech_stock_new_original,
            retrofit_potential=retrofit_potential, retrofit_cost=retrofit_cost, investment_potential=investment_potential,
            activity_gap_drivers=activity_gap, investments_drivers=investments_drivers, investments_nerghg=investments_nerghg,
            investments_ccus=investments_ccus, investments_energy=investments_energy, investments_power=investments_power,
            preliminary_investments=preliminary_investments, forced_decommissionings=forced_decommissionings,
            tech_stock_after_all=tech_stock_new))

    return activities, technologies


def _dump_invest_state(out_dir, activities, technologies, iP, arrays):
    """Debug only (settings debug_dump): save this period's intermediate investment arrays so two runs can be compared array by array."""
    import os
    extra = {}
    try:
        _pe = activities.energies.prices
        if iP == 0 and getattr(_pe, 'prev', None) is not None:
            extra['prices_used_energy'] = _pe.prev.yearly
            extra['prices_used_emission'] = activities.emissions.prices.prev.yearly
            extra['price_ranges_used'] = _pe.prev.ranges
        elif iP == 0:
            extra['prices_used_energy'] = _pe.initialized
            extra['prices_used_emission'] = activities.emissions.prices.initialized
        else:
            extra['prices_used_energy'] = _pe.yearly[:, iP - 1]
            extra['prices_used_emission'] = activities.emissions.prices.yearly[:, iP - 1]
            extra['price_ranges_used'] = _pe.ranges[:, :, iP - 1]
        extra['activity_volumes'] = activities.volumes[:, iP]
        extra['lcop_values'] = technologies.balancers.lcops.values[:, iP]
        extra['mca_matrix'] = technologies.balancers.mca.matrix[:, :, iP]
        extra['stock_max'] = technologies.balancers.stocks.max[:, iP]
        extra['stock_min'] = technologies.balancers.stocks.min[:, iP]
        extra['investment_costs'] = technologies.balancers.costs.investments[:, iP]
    except Exception as e:
        extra['_error'] = np.array([str(e)])
    out = {}
    for k, v in {**arrays, **extra}.items():
        try:
            out[k] = np.asarray(v, dtype=float)
        except Exception:
            pass
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"debug_invest_{int(activities.periods[iP])}.npz"), **out)