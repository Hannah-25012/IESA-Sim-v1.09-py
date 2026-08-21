"""
Tier-1 Sim->Opt hand-off: seed an Opt input DB with IESA-Sim's realized stock
at a hand-off year, so Opt optimizes the remaining path (e.g. 2030 -> 2050) on
top of the fleet Sim actually built. See the "2030 Handoff" design note.

This is the single implementation shared by:
  - scratch/extract_tier1.py  (the standalone CLI)
  - app.py  POST /handoff/extract  (the run-handoff.html screen)

Pure-stdlib + duckdb; no IESA-Sim project imports, so it stays importable from
both the API process and a bare `python scratch/extract_tier1.py`.

Data path (all names live in the merged / Sim-output schemas):
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
