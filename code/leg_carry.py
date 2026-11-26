"""
Leg mode: let a Sim run that starts part-way through the horizon behave like the SAME run continuing.

A continuous run's period p reads, besides the previous period's stock, three things the previous period produced:
  - the yearly energy prices, the emission prices and the price-duration ranges (invest_tech_LCOPs.py reads [iP-1]),
  - each generator's capture rate, normalised utilisation and cash flow (invest_power_technologies.py reads [iP-1]).
At iP == 0 there is no previous period, so the model uses the initial prices, skips power investment and leaves out the flexibility
technologies - correct for the true first period (the calibrated initial year), wrong for a leg that continues a finished run.

handoff_tier1.seed_sim_from_sim(..., prev=P) writes the previous run's values at period P into four leg_carry_* tables of the seeded
database. With settings['leg_mode'] this module reads them and attaches them as `.prev` on the structures that hold the per-period
arrays, and the three sites above use `.prev` at iP == 0 instead of the initial values. Without leg_mode nothing here runs and the
model is unchanged.
"""
import duckdb
import numpy as np

from datastructures import Struct

CARRY_TABLES = ("leg_carry_energy_prices", "leg_carry_energy_ranges", "leg_carry_emission_prices", "leg_carry_generators")


def _vector(rows, names, what):
    got = dict(rows)
    missing = [n for n in names if n not in got]
    if missing:
        raise ValueError(f"leg_mode: {len(missing)} {what} missing from the carried state, e.g. {missing[:3]}")
    return np.array([got[n] for n in names], dtype=float)


def load_leg_carry(db_path, technologies, activities):
    """Attach the previous period's state to the model structures (see the module docstring). Raises if the seeded DB has no carry tables."""
    con = duckdb.connect(db_path, read_only=True)
    try:
        have = {r[0] for r in con.execute("show tables").fetchall()}
        absent = [t for t in CARRY_TABLES if t not in have]
        if absent:
            raise ValueError(f"leg_mode needs {absent} in {db_path}; seed the leg with handoff sim_continue(prev=...) from a Sim run that saved them")
        energy_names = list(activities.energies.names)
        emission_names = list(activities.emissions.names)
        tech_ids = list(technologies.balancers.ids)

        yearly = _vector(con.execute("select activity_name, price from leg_carry_energy_prices").fetchall(), energy_names, "energy activities")
        emis = _vector(con.execute("select activity_name, price from leg_carry_emission_prices").fetchall(), emission_names, "emission activities")

        nRp = activities.energies.prices.ranges.shape[0]
        ranges = np.zeros((nRp, len(energy_names)))
        idx = {n: i for i, n in enumerate(energy_names)}
        seen = set()
        for name, r, price in con.execute("select activity_name, price_range_idx, price from leg_carry_energy_ranges").fetchall():
            if name in idx and 0 <= r < nRp:
                ranges[int(r), idx[name]] = price; seen.add(name)
        if len(seen) != len(energy_names):
            raise ValueError("leg_mode: price ranges carried for %d of %d energy activities" % (len(seen), len(energy_names)))

        gen = {n: (a, b, c) for n, a, b, c in con.execute("select tech_id, nuf, capture_rate, cash_flow from leg_carry_generators").fetchall()}
        # The DB stores an undefined indicator (no output to measure: a never-built generator) as NULL, while a continuous run holds NaN there,
        # and invest_power_technologies.py reads NaN as "passes the gate, first foothold allowed" but 0 as "fails". So NULL must come back as NaN
        # (not 0), or every technology without history is blocked in the leg's first period. Cash flow is never NULL (0 = no cash flow yet).
        nan = float("nan")
        def _col(k, null):
            return np.array([(gen[t][k] if gen[t][k] is not None else null) if t in gen else null for t in tech_ids], dtype=float)
        NUF, CR, CF = _col(0, nan), _col(1, nan), _col(2, 0.0)
    finally:
        con.close()

    activities.energies.prices.prev = Struct(yearly=yearly, ranges=ranges)
    activities.emissions.prices.prev = Struct(yearly=emis)
    technologies.balancers.generators.prev = Struct(NUF=NUF, CR=CR, CF=CF)
    return {"energy_activities": len(energy_names), "generators": len(gen)}
