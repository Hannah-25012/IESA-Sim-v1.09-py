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
                                "error": "..." | null}
  GET  /run/{job_id}/download -> zip of simulation_excel.duckdb + simulation_results.pkl
                                for the GUI's "save results" action (only once done;
                                simulation_state.duckdb, the multi-GB raw solver state,
                                is deliberately left out of this bundle)
  POST /checkFile           -> multipart file upload (.xlsx/.duckdb) -> a fast,
                                non-destructive shape probe (see code/read/compat_check.py),
                                not a full parse - for the unified-project gateway's
                                own compare/unify wizard to badge a dropped file
                                without paying for a real conversion
  GET  /run/{job_id}/graph/{name} -> one of the PNGs listed in the job's
                                meta.graphs, for the GUI to render inline
                                (the model itself only ever saves these to
                                disk now - see code/write/graph_*.py - so this
                                is the only way to see them; nothing pops up
                                as an on-screen window anymore, headless
                                container or not)

Only one simulation runs at a time (single-worker executor) - the model
mutates a handful of on-disk files (SIMmodel.duckdb, output/<scenario>/...)
that aren't safe to write to concurrently from two runs.
"""
import io
import os
import sys
import tempfile
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import duckdb
import matplotlib
matplotlib.use("Agg")  # headless container: no display to show interactive figures on

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

# Same shape/defaults as settings/IESA_settings_v1.10.json - callers only
# need to override the fields they care about.
_DEFAULT_SETTINGS = {
    "file_name": "IESA-Opt_v4.4.5 Policy + Agents.xlsx",
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
    }


def _summarize_output(scenario_name: str) -> dict:
    """Pull a compact JSON-safe summary from simulation_excel.duckdb rather
    than returning the full relational database (which includes large
    hourly tables) inline in the HTTP response."""
    db_path = Path("output") / scenario_name / "simulation_excel.duckdb"
    if not db_path.exists():
        return {}

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        def rows(sql):
            cols = [d[0] for d in con.execute(sql).description]
            return [dict(zip(cols, r)) for r in con.execute(sql).fetchall()]

        return {
            "system_costs": rows("SELECT * FROM system_costs ORDER BY period, cost_category"),
            "system_emissions": rows("SELECT * FROM system_emissions ORDER BY period"),
            "policy_cashflows": rows("SELECT * FROM policy_cashflows ORDER BY period, cashflow_category"),
        }
    finally:
        con.close()


def _list_graphs(scenario_name: str) -> list[str]:
    graphs_dir = Path("output") / scenario_name / "graphs"
    if not graphs_dir.is_dir():
        return []
    return sorted(p.name for p in graphs_dir.glob("*.png"))


def _run_job(job_id: str, settings: dict) -> None:
    _jobs[job_id]["status"] = "running"
    started = time.perf_counter()
    try:
        run_simulation(settings)
        _jobs[job_id].update(
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
        _jobs[job_id].update(status="error", error=f"{type(e).__name__}: {e}")


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


@app.get("/run/{job_id}/download")
def download_results(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"No job with id {job_id!r}")
    if job["status"] != "done":
        raise HTTPException(400, f"Job is {job['status']!r}, not done yet")

    scenario_name = job["meta"]["scenario_name"]
    out_dir = Path("output") / scenario_name

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name in _DOWNLOAD_FILES:
            path = out_dir / name
            if path.exists():
                zf.write(path, arcname=name)
    buf.seek(0)

    zip_name = f"{scenario_name}_results.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
    )


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
    return FileResponse(path, media_type="image/png")


@app.post("/run")
def run(req: RunRequest):
    settings = _build_settings(req.input)
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending", "output": None, "meta": None, "error": None}
    _executor.submit(_run_job, job_id, settings)
    return {"job_id": job_id}


@app.get("/run/{job_id}")
def run_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"No job with id {job_id!r}")
    return job
