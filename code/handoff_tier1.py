"""
Tier-1 Sim<->Opt hand-off: seed one engine's input with the other's realized
stock at a hand-off year, so the receiving engine continues on top of the
fleet the other actually built/chose. See the "2030 Handoff" design note.

This is the single implementation shared by:
  - scratch/extract_tier1.py  (the standalone CLI, forward direction only)
  - app.py  POST /handoff/extract   (Sim -> Opt, the run-handoff.html screen)
  - app.py  POST /handoff/reverse   (Opt -> Sim, the chained run modes)

Pure-stdlib + duckdb; no IESA-Sim/IESA-Opt project imports, so it stays
importable from both the API process and a bare `python scratch/extract_tier1.py`.

Data path, forward direction (Sim -> Opt; all names live in the merged /
Sim-output schemas):
  SOURCE  Sim run output DuckDB
      technology_stock            (tech_id, period, stock)         # per-run excel DB
      technology_stock_evolution  (tech_id, period, stock)         # full-state DB
      infrastructure_stock_evolution (infra_id, period, stock)     # full-state DB only
      technology_decommissionings (tech_id, period, decommissioning)  # full-state DB only
  BRIDGE  merged/Opt input DB
      technologies.sim_id  -> technologies.id (== opt_id namespace)
      infrastructure.sim_id -> infrastructure.id
  TARGET  a COPY of the merged/Opt input DB
      technologies.stock_initial   <- Sim stock @handoff       # Opt techStock_exist
      infrastructure.stock_initial  <- Sim infra stock @handoff
      technology_stocks.dec_planned <- Sim decommissioning, periods > handoff
      periods_solve.value           <- [handoff .. last period]

Data path, reverse direction (Opt -> Sim; see seed_from_opt):
  SOURCE  Opt results DuckDB (results.duckdb)
      techStock  (tech, period, value)   # dense, keyed by Opt's own tech id -
                                          # the SAME namespace as merged
                                          # technologies.id, so no sim_id
                                          # bridge or unit conversion needed.
                                          # julia-backend's ModelSets.technologies
                                          # is tech_balancers UNION tech_infra
                                          # (sets.jl:365-376) and infra techs
                                          # share the SAME techStock JuMP
                                          # variable (model/infrastructure.jl),
                                          # so this one table already carries
                                          # infrastructure rows too - under
                                          # infrastructure.id, not technologies.id.
  TARGET  a COPY of the merged/Sim input DB, sliced to a two-period leg
      technologies.stock_initial   <- Opt stock @start          # Sim iP==0 seed
      infrastructure.stock_initial <- Opt stock @start          # same techStock
                                        table, joined against infrastructure.id
                                        instead - see SOURCE note above
      technology_stocks.dec_planned <- NULL for periods <= start (already
                                        reflected in the seeded stock; avoids
                                        Opt's own first-period decom_plannedSel
                                        re-absorbing it on a later leg)
      periods                       <- [start, harvest] only, period_order
                                        renumbered so `start` becomes Sim's
                                        own iP==0 (see mod0_load_duckdb.py:139
                                        and main.py's period loop)
      energy_types."price init"    <- (optional, feed_prices=True) Opt's
                                        activity_prices @start, per energy
                                        label - the only lever that reaches
                                        Sim's next leg (see invest_tech_LCOPs.py,
                                        consumed only at iP==0)
"""
import os
import shutil

import duckdb


def pick_stock_table(sim_tables: set) -> str:
    # per-run excel DB exposes `technology_stock`; full-state DB `technology_stock_evolution`.
    for name in ("technology_stock", "technology_stock_evolution"):
        if name in sim_tables:
            return name
    raise ValueError("Sim DB has neither technology_stock nor technology_stock_evolution.")


def seed_from_sim(sim_db: str, merged_db: str, out_db: str, handoff: int) -> dict:
    """Write `out_db` = `merged_db` seeded with Sim's stock at `handoff`.

    Never mutates `merged_db` (works on a copy). Returns a structured report;
    raises ValueError on a fatal precondition (missing/all-zero hand-off year).
    """
    for p in (sim_db, merged_db):
        if not os.path.isfile(p):
            raise ValueError(f"not found: {p}")

    os.makedirs(os.path.dirname(os.path.abspath(out_db)), exist_ok=True)
    shutil.copyfile(merged_db, out_db)
    con = duckdb.connect(out_db)                       # read-write target
    try:
        con.execute(f"ATTACH '{sim_db}' AS sim (READ_ONLY)")
        sim_tables = {r[0] for r in con.execute("show tables from sim").fetchall()}
        stock_tbl = pick_stock_table(sim_tables)
        H = int(handoff)

        # --- guards ---------------------------------------------------------
        sim_years = [r[0] for r in con.execute(
            f"select distinct period from sim.{stock_tbl} order by period").fetchall()]
        if H not in sim_years:
            raise ValueError(f"hand-off {H} not in Sim periods {sim_years}.")
        # Sim zero-fills stock for periods beyond its year_end; an all-zero
        # column at H means Sim never simulated that far -> seeding would wipe
        # Opt's fleet.
        handoff_total = con.execute(
            f"select coalesce(sum(abs(stock)), 0) from sim.{stock_tbl} where period = {H}").fetchone()[0]
        if handoff_total == 0:
            raise ValueError(
                f"Sim stock at {H} is all-zero - this run's year_end is before {H}. "
                f"Re-run Sim with year_end >= {H} before extracting.")

        # --- coverage / hazard report (computed before the writes) ----------
        matched = con.execute(f"""
            select count(*) from technologies t
            join sim.{stock_tbl} s on s.tech_id = t.sim_id and s.period = {H}
        """).fetchone()[0]
        sim_only = con.execute(f"""
            select count(distinct s.tech_id) from sim.{stock_tbl} s
            where s.period = {H}
              and s.tech_id not in (select sim_id from technologies where sim_id is not null)
        """).fetchone()[0]
        dupes = con.execute("""
            select sim_id, count(*) c from technologies
            where sim_id is not null group by sim_id having count(*) > 1
        """).fetchall()
        unit_mismatch = con.execute(f"""
            select count(*) from technologies t
            join sim.technologies st on st.id = t.sim_id
            where t.unit is not null and st.unit is not null and t.unit <> st.unit
              and t.sim_id in (select tech_id from sim.{stock_tbl} where period = {H})
        """).fetchone()[0]

        # --- TIER 1 core: seed existing stock from Sim @handoff -------------
        con.execute(f"""
            update technologies t set stock_initial = s.stock
            from sim.{stock_tbl} s
            where s.tech_id = t.sim_id and s.period = {H}
        """)

        # --- TIER 1 honesty: infra stock + retirement schedule (state DB) ---
        infra_seeded = 0
        if "infrastructure_stock_evolution" in sim_tables:
            con.execute(f"""
                update infrastructure i set stock_initial = s.stock
                from sim.infrastructure_stock_evolution s
                where s.infra_id = i.sim_id and s.period = {H}
            """)
            infra_seeded = con.execute(f"""
                select count(*) from infrastructure i
                join sim.infrastructure_stock_evolution s
                  on s.infra_id = i.sim_id and s.period = {H}
            """).fetchone()[0]

        decom_rows = 0
        if "technology_decommissionings" in sim_tables:
            con.execute(f"""
                update technology_stocks ts set dec_planned = d.decommissioning
                from sim.technology_decommissionings d, technologies t
                where t.sim_id = d.tech_id and ts.tech_id = t.id
                  and ts.period = d.period and d.period > {H}
            """)
            decom_rows = con.execute(
                f"select count(*) from sim.technology_decommissionings where period > {H}").fetchone()[0]

        # --- set Opt's solve window to start at the hand-off ---------------
        con.execute("delete from periods_solve")
        con.execute(
            f"insert into periods_solve (value) select period from periods where period >= {H} order by period")
        solve_window = [r[0] for r in con.execute("select value from periods_solve order by value").fetchall()]

        return {
            "handoff": H,
            "sim_stock_table": stock_tbl,
            "seeded_technologies": matched,
            "seeded_infrastructure": infra_seeded,
            "dec_planned_rows": decom_rows,
            "solve_window": solve_window,
            "sim_only_ignored": sim_only,
            "unit_mismatches": unit_mismatch,
            "duplicate_sim_ids": [{"sim_id": r[0], "opt_rows": r[1]} for r in dupes],
            "full_state_source": "infrastructure_stock_evolution" in sim_tables,
        }
    finally:
        con.close()


def seed_from_opt(opt_results_db: str, merged_db: str, out_db: str, start: int, harvest: int,
                   feed_prices: bool = False) -> dict:
    """Write `out_db` = `merged_db` seeded with Opt's realized stock at
    `start`, sliced to the two-period leg [start, harvest] IESA-Sim needs to
    warm-start from `start` and build real power capacity by `harvest` (a
    one-period leg builds none - see mod2_invest.py's `if iP > 0:` gate on
    power-generation investment).

    Never mutates `merged_db` (works on a copy). Returns a structured report;
    raises ValueError on a fatal precondition (missing/all-zero start year,
    start/harvest not both real periods of the merged DB).
    """
    for p in (opt_results_db, merged_db):
        if not os.path.isfile(p):
            raise ValueError(f"not found: {p}")
    S, H = int(start), int(harvest)
    if S >= H:
        raise ValueError(f"start ({S}) must be before harvest ({H}).")

    os.makedirs(os.path.dirname(os.path.abspath(out_db)), exist_ok=True)
    shutil.copyfile(merged_db, out_db)
    con = duckdb.connect(out_db)                       # read-write target
    try:
        con.execute(f"ATTACH '{opt_results_db}' AS opt_res (READ_ONLY)")
        opt_tables = {r[0] for r in con.execute("show tables from opt_res").fetchall()}
        if "techStock" not in opt_tables:
            raise ValueError("Opt results DB has no techStock table.")

        # --- guards ---------------------------------------------------------
        opt_periods = [r[0] for r in con.execute(
            "select distinct period from opt_res.techStock order by period").fetchall()]
        if S not in opt_periods:
            raise ValueError(f"start {S} not in Opt's solved periods {opt_periods}.")
        start_total = con.execute(
            f"select coalesce(sum(abs(value)), 0) from opt_res.techStock where period = {S}").fetchone()[0]
        if start_total == 0:
            raise ValueError(f"Opt stock at {S} is all-zero - Opt did not solve that period "
                             f"(or every technology is genuinely at zero capacity there).")
        db_periods = [r[0] for r in con.execute("select period from periods order by period").fetchall()]
        missing = [y for y in (S, H) if y not in db_periods]
        if missing:
            raise ValueError(f"period(s) {missing} not in the merged DB's own periods {db_periods}.")

        # --- coverage report (computed before the writes) --------------------
        # Opt's tech id IS the merged DB's technologies.id (both Opt-anchored -
        # see the module docstring), so this join needs no sim_id bridge and
        # no unit check, unlike the forward direction. Opt's techStock also
        # carries infrastructure.id rows in the same table (see module
        # docstring) - a second join against infrastructure covers those.
        matched = con.execute(f"""
            select count(*) from technologies t
            join opt_res.techStock s on s.tech = t.id and s.period = {S}
        """).fetchone()[0]
        infra_matched = con.execute(f"""
            select count(*) from infrastructure i
            join opt_res.techStock s on s.tech = i.id and s.period = {S}
        """).fetchone()[0]
        opt_only = con.execute(f"""
            select count(distinct s.tech) from opt_res.techStock s
            where s.period = {S}
              and s.tech not in (select id from technologies)
              and s.tech not in (select id from infrastructure)
        """).fetchone()[0]

        # --- TIER 1 core: seed existing stock from Opt @start ----------------
        con.execute(f"""
            update technologies t set stock_initial = s.value
            from opt_res.techStock s
            where s.tech = t.id and s.period = {S}
        """)
        con.execute(f"""
            update infrastructure i set stock_initial = s.value
            from opt_res.techStock s
            where s.tech = i.id and s.period = {S}
        """)

        # --- re-base planned decommissioning: any event at/before `start` is
        # already reflected in the seeded stock (mirrors seed_from_sim's own
        # "period > H" convention above, going the other way) - leaving it in
        # would have Opt's own first-solved-period decom_plannedSel absorption
        # (parameters.jl's lower_excl = typemin(Int)) re-retire it a second
        # time on a later leg. Sim itself doesn't read this column (only Opt
        # does), so clearing it here is harmless to the upcoming Sim leg and
        # only matters for whatever Opt leg reads this DB next. ---
        decom_cleared = con.execute(
            f"select count(*) from technology_stocks where dec_planned is not null and period <= {S}"
        ).fetchone()[0]
        con.execute(f"update technology_stocks set dec_planned = NULL where period <= {S}")

        # --- slice the period grid to this leg: IESA-Sim takes its period
        # vector straight from this table (mod0_load_duckdb.py:139), ordered
        # by period_order, and reindexes every per-period array against it -
        # so restricting to [S, H] and renumbering makes S become iP==0. ---
        con.execute(f"delete from periods where period < {S} or period > {H}")
        con.execute("""
            update periods set period_order = sub.new_order
            from (select period, row_number() over (order by period) - 1 as new_order
                  from periods) sub
            where periods.period = sub.period
        """)
        period_grid = [r[0] for r in con.execute(
            "select period from periods order by period_order").fetchall()]

        # --- optional price feedback (bidirectional chained mode): Sim has no
        # input table for realized per-period prices, only a per-energy-label
        # initial-price scalar consumed exclusively at iP==0
        # (energy_types."price init" -> mod0_load_duckdb.py:101-102 ->
        # invest_tech_LCOPs.py's iP==0 branch) - the one lever that actually
        # reaches Sim's next leg. ---
        prices_fed = 0
        if feed_prices and "activity_prices" in opt_tables:
            has_energy_types = con.execute(
                "select count(*) from information_schema.tables where table_name = 'energy_types'"
            ).fetchone()[0] > 0
            if has_energy_types:
                con.execute(f"""
                    update energy_types et set "price init" = p.price
                    from opt_res.activity_prices p
                    join activities a on a."Name" = p.activity
                    where a.energy_label = et.labels and p.period = {S}
                """)
                prices_fed = con.execute(f"""
                    select count(distinct et.labels) from energy_types et
                    join activities a on a.energy_label = et.labels
                    join opt_res.activity_prices p on p.activity = a."Name" and p.period = {S}
                """).fetchone()[0]

        return {
            "start": S,
            "harvest": H,
            "seeded_technologies": matched,
            "seeded_infrastructure": infra_matched,
            "opt_only_ignored": opt_only,
            "dec_planned_cleared": decom_cleared,
            "period_grid": period_grid,
            "prices_fed": prices_fed,
            "infrastructure_carried": infra_matched > 0,
        }
    finally:
        con.close()
