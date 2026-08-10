"""FastAPI wrapper so this project can be called from the unified-project
gateway (see D:\\I-AI\\UniversityCode\\unified-project\\templates\\python-backend).

A full IESA-Sim run can take 10+ minutes, which is longer than the
gateway's default proxy timeout and longer than a single HTTP request
should reasonably block for - so /run deviates from the template's
strictly-synchronous contract and uses a small async job pattern instead
(the unified-project's dbcompare-backend already establishes that custom
endpoints are fine for a backend with its own domain shape):

  GET  /health              -> liveness check
  POST /input               -> multipart file upload (.xlsx) -> {"file_name": "..."},
                                saved to input/ so a run can reference it
  POST /run                 -> {"input": {...}} -> {"job_id": "..."}, starts a
                                background run and returns immediately
  GET  /run/{job_id}        -> {"status": "pending"|"running"|"done"|"error",
                                "output": {...} | null, "meta": {...} | null,
                                "error": "..." | null, "logs": ["...", ...]}
                                - logs is every line main.py has print()'d so
                                far (grows during "running"), so the GUI can
                                show the same progress the console already
                                gets instead of just a bare status word
                                across the whole 10+ minute run
  GET  /run/{job_id}/download -> zip of simulation_excel.duckdb + simulation_results.pkl
                                for the GUI's "save results" action (only once done;
                                simulation_state.duckdb, the multi-GB raw solver state,
                                is deliberately left out of this bundle)
  POST /checkFile           -> multipart file upload (.xlsx/.duckdb) -> a fast,
                                non-destructive shape probe (see code/read/compat_check.py),
                                not a full parse - for the unified-project gateway's
                                own compare/unify wizard to badge a dropped file
                                without paying for a real conversion
  GET  /run/{job_id}/graph/{name} -> one of the interactive Plotly HTML
                                charts listed in the job's meta.graphs, for
                                the GUI to embed inline (via <iframe> - see
                                code/write/graph_*.py, which write these
                                straight to disk instead of popping up an
                                on-screen window; this is the only way to
                                see them)
  GET  /outputs             -> [{"scenario_name", "graphs", "modified",
                                "has_results"}, ...] for every output/<scenario>
                                directory that has saved graphs, newest first -
                                lets the GUI offer "load a previous run" without
                                needing the in-memory job that produced it (jobs -
                                and their job_id - don't survive a container
                                restart, but the saved output/ files do).
                                has_results is false when a run has graphs but
                                crashed/was killed before simulation_excel.duckdb
                                was written (graphs write first) - /results would
                                silently return {} for such a scenario
  GET  /outputs/{scenario_name}/results -> same shape as a job's own "output"
                                (system_costs/system_emissions/policy_cashflows);
                                pass ?series=sectoral_emissions,technology_stock,
                                technology_use,prices for multi-scenario compare
  GET  /outputs/{scenario_name}/graph/{name} -> same as the job-scoped graph
                                route above, keyed by scenario name instead
  GET  /outputs/{scenario_name}/download -> same as /run/{job_id}/download,
                                keyed by scenario name instead - the GUI's
                                "save results" needs this too once results
                                came from /outputs (loaded, no live job_id)
                                rather than a just-finished run
  DELETE /outputs/{scenario_name} -> removes output/<scenario_name>/ plus its
                                sidecar Excel reports (<scenario_name>_general.xlsx
                                etc., written alongside it by results_write.py)
                                -> {"deleted": "<scenario_name>"}

Only one simulation runs at a time (single-worker executor) - the model
mutates a handful of on-disk files (SIMmodel.duckdb, output/<scenario>/...)
that aren't safe to write to concurrently from two runs.
"""
import io
import os
import shutil
import sys
import tempfile
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import duckdb

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

_code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code")
if _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)

from main import main as run_simulation  # noqa: E402  (path setup must run first; also puts code/read/ on sys.path)
from mod0_read_data_save_duck import mod0_read_data_save_duck  # noqa: E402
from compat_check import check_sim_compatibility  # noqa: E402

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, dict[str, Any]] = {}


class _TeeCapture:
    """Mirrors every write to `real_stream` (so the container's own console/
    `docker logs` still sees everything main.py print()s, unchanged) while
    also line-buffering it into `job["logs"]`, so a GUI polling GET /run/{id}
    mid-run sees the same progress the console already got instead of just
    a bare "running" status for the whole 10+ minute run. sys.stdout/stderr
    are process-global, not per-thread, but _executor's single worker means
    only one job's run_simulation() is ever mid-flight at a time - no risk of
    two runs' output interleaving into the wrong job's log."""

    def __init__(self, real_stream, job: dict):
        self._real = real_stream
        self._job = job
        self._buf = ""

    def write(self, s):
        self._real.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._job["logs"].append(line)

    def flush(self):
        self._real.flush()

# Same shape/defaults as settings/IESA_settings_v1.10.json - callers only
# need to override the fields they care about.
_DEFAULT_SETTINGS = {
    # The plain "...Policy + Agents.xlsx" (no suffix) is kept exactly as
    # originally shipped - it does NOT have the emission-sign flip below, so
    # it is no longer a valid default on its own (see next paragraph).
    #
    # code/write/results_emissions.py and friends have unconditionally
    # expected IESA-Opt's positive-for-emitters sign convention since
    # "Invert emission coefficient sign convention to match IESA-Opt" - that
    # is not optional/file-dependent, so the default input file MUST be one
    # of the "...(emissions sign flipped)..." variants, or every emissions-
    # derived output (system_emissions, sectoral_emissions, the emissions
    # graph, EUA/tax cashflows, ...) comes out sign-inverted across the
    # board (every sector strongly negative every period - caught only by
    # eyeballing the system emissions graph after this file briefly pointed
    # at a non-flipped variant; the negative numbers were not otherwise
    # obviously wrong at a glance).
    #
    # This default also carries the corrected Alkaline Electrolyzer/Imported
    # Hydrogen (liquid) names (pure sharedStrings.xml text, verified to
    # produce numerically identical output on top of a sign-matched base)
    # and IESA-Opt's own shed_capacity derivation (shedding_capacity_percentage
    # * peak(hourly_profile) * cap2act, see disp_initialize_power.py and
    # julia-backend/src/parameters.jl's compute_shed_capacity!, instead of
    # treating shedding_capacity as an already-final per-unit-capacity rate)
    # - shedding_capacity in the workbook is now a plain [%] matching
    # IESA-Opt's convention, a real behavior change for the 6 technologies
    # that have one, not a relabeling.
    "file_name": "IESA-Opt_v4.4.5 Policy + Agents (emissions sign flipped, fixed names, opt shedding logic).xlsx",
    "scenario_name": "Policy + Agents",
    "db_path": "SIMmodel.duckdb",
    # Defaults to True so a freshly-built container (no duckDB baked in,
    # see .dockerignore) is self-sufficient on its first run. Pass
    # read_input=false for faster subsequent runs once db_path already
    # exists (e.g. on a mounted volume that persists across restarts).
    "read_input": True,
    "plot_price_duration": False,
    "nIp": 1,
    "nIb": 10,
    "nId": 2,
    "year_end": 2050,
    # simulation_state.duckdb (the full 8760-hour-resolution solver-state
    # dump) is write-only from this API's perspective - nothing here ever
    # reads it back, it's deliberately excluded from the download zip below,
    # and building it has been enough to OOM-kill the container mid-write
    # (see main.py). Off by default; pass true to opt back in for offline
    # analysis outside this API.
    "save_state_duckdb": False,
}


class RunRequest(BaseModel):
    input: Optional[dict] = None


def _build_settings(overrides: Optional[dict]) -> dict:
    merged = dict(_DEFAULT_SETTINGS)
    if overrides:
        unknown = set(overrides) - set(_DEFAULT_SETTINGS)
        if unknown:
            raise HTTPException(400, f"Unknown input field(s): {sorted(unknown)}")
        merged.update(overrides)

    return {
        "input": os.path.join("input", merged["file_name"]),
        "scenario_name": merged["scenario_name"],
        "db_path": merged["db_path"],
        "read_input": merged["read_input"],
        "save_output": True,  # forced on: /run's output summary is read back from it
        "plot_price_duration": merged["plot_price_duration"],
        "iterations": {
            "power": merged["nIp"],
            "balancing": merged["nIb"],
            "dispatch": merged["nId"],
        },
        "year_end": merged["year_end"],
        "save_state_duckdb": merged["save_state_duckdb"],
    }


# Extra series _summarize_output can be asked to include on top of its
# always-on 3 (see GET /outputs/{name}/results' `series` query param) - kept
# out of the default payload since live job results (_run_job) and the
# existing single-scenario Load path only ever need the cheap 3, while
# multi-scenario comparison (run.html's Compare flow) wants more per call.
_EXTRA_OUTPUT_SERIES = {"sectoral_emissions", "technology_stock", "technology_use", "prices"}


def _summarize_output(scenario_name: str, extra: frozenset[str] = frozenset()) -> dict:
    """Pull a compact JSON-safe summary from simulation_excel.duckdb rather
    than returning the full relational database (which includes large
    hourly tables) inline in the HTTP response. `extra` opts into additional
    series beyond the always-on 3 - see _EXTRA_OUTPUT_SERIES."""
    db_path = Path("output") / scenario_name / "simulation_excel.duckdb"
    if not db_path.exists():
        return {}

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        def rows(sql):
            cols = [d[0] for d in con.execute(sql).description]
            return [dict(zip(cols, r)) for r in con.execute(sql).fetchall()]

        result = {
            "system_costs": rows("SELECT * FROM system_costs ORDER BY period, cost_category"),
            "system_emissions": rows("SELECT * FROM system_emissions ORDER BY period"),
            "policy_cashflows": rows("SELECT * FROM policy_cashflows ORDER BY period, cashflow_category"),
        }

        if "sectoral_emissions" in extra:
            result["sectoral_emissions"] = rows(
                "SELECT sector, period, positive, negative FROM sectoral_emissions ORDER BY sector, period")

        # technology_stock/use are per-technology (not pre-aggregated) - the
        # same granularity compare-results.html already sends for Sim vs Opt,
        # aggregated client-side by sector/category rather than in SQL here.
        if "technology_stock" in extra:
            result["technology_stock"] = rows("""
                SELECT t.name, t.sector, t.category, ts.period, ts.stock
                FROM technology_stock ts JOIN technologies t ON t.id = ts.tech_id
                ORDER BY t.sector, t.name, ts.period""")

        if "technology_use" in extra:
            result["technology_use"] = rows("""
                SELECT t.name, t.sector, t.category, tu.period, tu.use
                FROM technology_use tu JOIN technologies t ON t.id = tu.tech_id
                ORDER BY t.sector, t.name, tu.period""")

        if "prices" in extra:
            result["energy_prices"] = rows(
                "SELECT activity_name, period, price FROM energy_prices ORDER BY activity_name, period")
            result["emission_prices"] = rows(
                "SELECT activity_name, period, price FROM emission_prices ORDER BY activity_name, period")

        return result
    finally:
        con.close()


def _list_graphs(scenario_name: str) -> list[str]:
    graphs_dir = Path("output") / scenario_name / "graphs"
    if not graphs_dir.is_dir():
        return []
    return sorted(p.name for p in graphs_dir.glob("*.html"))


def _list_output_scenarios() -> list[dict]:
    """Every output/<scenario> directory that actually has saved graphs - the
    same on-disk source of truth a job's own /run/{job_id} results already
    read from, but keyed by scenario name directly instead of a job id, so a
    past run's results/graphs stay browsable even after the in-memory job
    that produced them is gone (a container restart wipes _jobs entirely;
    the output/ files on disk don't move). Computed fresh on every call
    rather than cached, so a run that just finished shows up immediately."""
    root = Path("output")
    if not root.is_dir():
        return []
    scenarios = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        graphs = _list_graphs(d.name)
        if not graphs:
            continue
        mtime = max((d / "graphs" / g).stat().st_mtime for g in graphs)
        scenarios.append({
            "scenario_name": d.name,
            "graphs": graphs,
            "modified": datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
            # Graphs are written before simulation_excel.duckdb (see
            # code/main.py), so a run that crashed or was killed partway
            # through can leave a scenario with graphs but no queryable
            # results - _summarize_output would silently return {} for it.
            # Flagged here so callers (run.html's Compare flow) can warn
            # instead of rendering that scenario as blank data.
            "has_results": (d / "simulation_excel.duckdb").exists(),
        })
    scenarios.sort(key=lambda s: s["modified"], reverse=True)
    return scenarios


def _run_job(job_id: str, settings: dict) -> None:
    job = _jobs[job_id]
    job["status"] = "running"
    started = time.perf_counter()
    real_stdout, real_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeCapture(real_stdout, job)
    sys.stderr = _TeeCapture(real_stderr, job)
    try:
        run_simulation(settings)
        job.update(
            status="done",
            output=_summarize_output(settings["scenario_name"]),
            meta={
                "language": "python",
                "version": "1.0",
                "scenario_name": settings["scenario_name"],
                "year_end": settings["year_end"],
                "duration_seconds": round(time.perf_counter() - started, 1),
                "graphs": _list_graphs(settings["scenario_name"]),
            },
        )
    except Exception as e:  # noqa: BLE001 - surface any failure to the job status instead of crashing the worker
        job.update(status="error", error=f"{type(e).__name__}: {e}")
    finally:
        sys.stdout, sys.stderr = real_stdout, real_stderr


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/input")
async def upload_input(file: UploadFile = File(...)):
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith(".xlsx"):
        raise HTTPException(400, "Input file must be a .xlsx workbook")

    os.makedirs("input", exist_ok=True)
    dest = os.path.join("input", filename)
    with open(dest, "wb") as f:
        f.write(await file.read())

    return {"file_name": filename}


# Lets the unified-project gateway's own compare/unify wizard run IESA-Sim
# directly against the database it just merged, the same way /convert lets it
# skip a full simulation to just get a DuckDB - here the caller already has a
# full database (Sim's own output, IESA-Opt's, or a merged one) and only
# wants /run to load it (db_path, read_input=false), not re-derive it from
# Excel. Separate from /input (which only ever takes the raw .xlsx workbook
# main.py itself would read) since the destination path and downstream use
# differ.
_UPLOADED_DB_DIR = "uploaded_dbs"


@app.post("/input_db")
async def upload_input_db(file: UploadFile = File(...)):
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith((".duckdb", ".db")):
        raise HTTPException(400, "Input file must be a .duckdb database")

    os.makedirs(_UPLOADED_DB_DIR, exist_ok=True)
    dest = os.path.join(_UPLOADED_DB_DIR, f"{uuid.uuid4()}_{filename}")
    with open(dest, "wb") as f:
        f.write(await file.read())

    return {"db_path": dest}


# Excel -> DuckDB conversion only, no solve. Separate from /run's job dict
# (different shape: no scenario_name/output dir) and from /input (which just
# saves bytes for /run to read later) — this is for the unified-project
# gateway's own compare/unify wizard, which needs a DuckDB out of a raw IESA-
# Sim workbook without paying for a full simulation.
_CONVERT_DIR = "converted"
_convert_jobs: dict[str, dict[str, Any]] = {}


def _convert_job(job_id: str, xlsx_path: str, db_path: str) -> None:
    _convert_jobs[job_id]["status"] = "running"
    try:
        mod0_read_data_save_duck(xlsx_path, db_path)
        _convert_jobs[job_id].update(status="done", outputPath=db_path, error=None)
    except Exception as e:  # noqa: BLE001 - surface any parse failure as a job error instead of crashing the worker
        _convert_jobs[job_id].update(status="error", error=f"{type(e).__name__}: {e}")


@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith((".xlsx", ".xlsm", ".xls")):
        raise HTTPException(400, "Input file must be an Excel workbook (.xlsx/.xlsm/.xls)")

    os.makedirs(_CONVERT_DIR, exist_ok=True)
    job_id = str(uuid.uuid4())
    xlsx_path = os.path.join(_CONVERT_DIR, f"{job_id}_{filename}")
    with open(xlsx_path, "wb") as f:
        f.write(await file.read())

    db_path = os.path.join(_CONVERT_DIR, f"{job_id}.duckdb")
    _convert_jobs[job_id] = {"status": "pending", "outputPath": None, "error": None}
    _executor.submit(_convert_job, job_id, xlsx_path, db_path)
    return {"job_id": job_id}


@app.post("/checkFile")
async def check_file(file: UploadFile = File(...)):
    filename = os.path.basename(file.filename or "")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".xlsx", ".xlsm", ".xls", ".duckdb", ".db"):
        raise HTTPException(400, f"Unsupported file type {ext!r} for compatibility check")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        return check_sim_compatibility(tmp_path)
    finally:
        os.remove(tmp_path)


@app.get("/convert/{job_id}")
def convert_status(job_id: str):
    job = _convert_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"No conversion job with id {job_id!r}")
    return job


@app.get("/convert/{job_id}/download")
def convert_download(job_id: str):
    job = _convert_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"No conversion job with id {job_id!r}")
    if job["status"] != "done":
        raise HTTPException(400, f"Job is {job['status']!r}, not done yet")

    db_path = job["outputPath"]

    def iter_file():
        with open(db_path, "rb") as f:
            yield from f

    return StreamingResponse(
        iter_file(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="sim.duckdb"'},
    )


_DOWNLOAD_FILES = ["simulation_excel.duckdb", "simulation_results.pkl"]


def _zip_results(scenario_name: str) -> io.BytesIO:
    out_dir = Path("output") / scenario_name
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name in _DOWNLOAD_FILES:
            path = out_dir / name
            if path.exists():
                zf.write(path, arcname=name)
    buf.seek(0)
    return buf


@app.get("/run/{job_id}/download")
def download_results(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"No job with id {job_id!r}")
    if job["status"] != "done":
        raise HTTPException(400, f"Job is {job['status']!r}, not done yet")

    scenario_name = job["meta"]["scenario_name"]
    zip_name = f"{scenario_name}_results.zip"
    return StreamingResponse(
        _zip_results(scenario_name),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
    )


@app.get("/outputs/{scenario_name}/download")
def download_output(scenario_name: str):
    # Same membership check as the other /outputs/{scenario_name} routes -
    # rules out path traversal without needing to separately sanitize.
    if not any(s["scenario_name"] == scenario_name for s in _list_output_scenarios()):
        raise HTTPException(404, f"No saved output named {scenario_name!r}")

    zip_name = f"{scenario_name}_results.zip"
    return StreamingResponse(
        _zip_results(scenario_name),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
    )


@app.get("/outputs/{scenario_name}/download_excel_duckdb")
def download_output_excel_duckdb(scenario_name: str):
    """Raw simulation_excel.duckdb only, not the zip - the combined Sim/Opt
    results-comparison page uploads this directly to dbcompare-backend's
    POST /results/load, the same way IESA-Opt's own results.duckdb is
    uploaded there (see julia-backend's GET /api/outputs/download)."""
    if not any(s["scenario_name"] == scenario_name for s in _list_output_scenarios()):
        raise HTTPException(404, f"No saved output named {scenario_name!r}")

    path = Path("output") / scenario_name / "simulation_excel.duckdb"
    if not path.exists():
        raise HTTPException(404, f"No simulation_excel.duckdb saved for {scenario_name!r}")
    return FileResponse(path, media_type="application/octet-stream", filename="simulation_excel.duckdb")


@app.get("/run/{job_id}/graph/{name}")
def get_graph(job_id: str, name: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"No job with id {job_id!r}")
    if job["status"] != "done":
        raise HTTPException(400, f"Job is {job['status']!r}, not done yet")

    # name must be exactly one of the filenames the job itself reported -
    # rules out path traversal (../, absolute paths) without needing to
    # separately sanitize the string.
    if name not in job["meta"]["graphs"]:
        raise HTTPException(404, f"No graph named {name!r} for this job")

    path = Path("output") / job["meta"]["scenario_name"] / "graphs" / name
    return FileResponse(path, media_type="text/html")


@app.get("/outputs")
def list_outputs():
    return {"outputs": _list_output_scenarios()}


@app.get("/outputs/{scenario_name}/results")
def output_results(scenario_name: str, series: str = ""):
    if not any(s["scenario_name"] == scenario_name for s in _list_output_scenarios()):
        raise HTTPException(404, f"No saved output named {scenario_name!r}")
    requested = {s.strip() for s in series.split(",") if s.strip()}
    unknown = requested - _EXTRA_OUTPUT_SERIES
    if unknown:
        raise HTTPException(400, f"Unknown series {sorted(unknown)} - choose from {sorted(_EXTRA_OUTPUT_SERIES)}")
    return _summarize_output(scenario_name, extra=frozenset(requested))


@app.delete("/outputs/{scenario_name}")
def delete_output(scenario_name: str):
    if not any(s["scenario_name"] == scenario_name for s in _list_output_scenarios()):
        raise HTTPException(404, f"No saved output named {scenario_name!r}")
    shutil.rmtree(Path("output") / scenario_name, ignore_errors=True)
    # Sidecar Excel reports (results_write.py) are written as siblings of
    # output/<scenario_name>/, not inside it - e.g. "<scenario_name>_general.xlsx" -
    # so they need their own cleanup here.
    for suffix in ("_general.xlsx", "_gas_prices_hourly.xlsx", "_power_prices_hourly.xlsx"):
        (Path("output") / f"{scenario_name}{suffix}").unlink(missing_ok=True)
    return {"deleted": scenario_name}


@app.get("/outputs/{scenario_name}/graph/{name}")
def output_graph(scenario_name: str, name: str):
    match = next((s for s in _list_output_scenarios() if s["scenario_name"] == scenario_name), None)
    if match is None:
        raise HTTPException(404, f"No saved output named {scenario_name!r}")
    # name must be exactly one of the filenames this scenario actually has -
    # rules out path traversal (../, absolute paths) without needing to
    # separately sanitize the string.
    if name not in match["graphs"]:
        raise HTTPException(404, f"No graph named {name!r} for {scenario_name!r}")

    path = Path("output") / scenario_name / "graphs" / name
    return FileResponse(path, media_type="text/html")


@app.post("/run")
def run(req: RunRequest):
    settings = _build_settings(req.input)
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending", "output": None, "meta": None, "error": None, "logs": []}
    _executor.submit(_run_job, job_id, settings)
    return {"job_id": job_id}


@app.get("/run/{job_id}")
def run_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"No job with id {job_id!r}")
    return job
