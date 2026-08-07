# =============================================================================
# compat_check.py - fast, non-destructive shape probes for whether a file has
# the sheets/headers/tables IESA-Sim's real reader
# (mod0_read_data_save_duck.py / mod0_load_duckdb.py) needs, without doing a
# full parse.
#
# Checks shape only, not data validity - the real readers also coerce values
# and can throw for reasons unrelated to layout (a stray non-numeric cell,
# say). Header reads below use the exact same skiprows/header offsets (and
# the same flatten_header helper) as mod0_read_data_save_duck.py, not
# reimplemented guesses - pandas' MultiIndex header handling for merged
# group-label cells doesn't reduce to a simple forward-fill (it fills upper
# levels but marks genuinely-blank trailing cells "Unnamed: N_level_M"), so
# reusing the real reader's own header parsing is what keeps this correct.
# =============================================================================
import os

import duckdb
import pandas as pd

from Constants.Parameters import (
    Parameters, Activities, Agents, Types, Retrofitting, HourlyProfiles,
    PriceProfiles, Technologies, Infrastructure,
)
from mod0_read_data_save_duck import flatten_header

# The exact tables mod0_load_duckdb.py's load_from_duckdb queries when
# loading a converted database back in for a run. Excludes
# "parameter_categories", which mod0_read_data_save_duck.py writes but the
# loader never reads.
_SIM_DUCKDB_TABLES = [
    "parameters", "activity_types", "sectors", "energy_types",
    "agent_types", "agent_profiles", "agent_criteria", "population",
    "criteria_weights", "periods", "activities", "hourly_profile_types",
    "hourly_profiles", "interconnectors", "price_profiles", "technologies",
    "technology_flexibility_activities", "technology_costs",
    "technology_stocks", "energy_balance", "infrastructure",
    "infrastructure_costs", "retrofittings", "policy_taxes",
    "policy_feedins", "policy_subsidies",
]


def _check_headers(missing, sheet, headers, required=(), prefixes=()):
    for name in required:
        if name not in headers:
            missing.append(f"{sheet}: {name}")
    for name in prefixes:
        if not any(str(h).startswith(name) for h in headers):
            missing.append(f"{sheet}: {name}")


def check_sim_excel_compatibility(path: str) -> dict:
    missing_sheets = []
    missing_headers = []
    try:
        xls = pd.ExcelFile(path)
    except Exception as e:  # noqa: BLE001 - not a real workbook at all; report, don't crash the endpoint
        return {"compatible": False, "missingSheets": [f"<could not open as an Excel workbook: {e}>"],
                "missingHeaders": []}
    try:
        sheetnames = set(xls.sheet_names)

        def need(name):
            return name in sheetnames

        # Header row is Excel row 3 (skiprows=2); required names are DATA
        # values in the "Name" column below it, not header text.
        if need("Parameters"):
            df = pd.read_excel(xls, sheet_name="Parameters", skiprows=2, nrows=100)
            if "Name" in df.columns:
                values = set(df["Name"].dropna().astype(str).str.strip())
                for n in (Parameters.powinv_SPBT_benchmark, Parameters.powinv_SPBT_min,
                          Parameters.powinv_CR_threshold, Parameters.powinv_CR_min,
                          Parameters.powinv_NUF_threshold, Parameters.powinv_NUF_min,
                          Parameters.scarcity_penalization, Parameters.gas_premium,
                          Parameters.voll_value, Parameters.min_spread_value,
                          Parameters.gov_dr, Parameters.exports_value):
                    if n not in values:
                        missing_headers.append(f"Parameters: {n}")
            else:
                missing_headers.append("Parameters: Name (column header, row 3)")
        else:
            missing_sheets.append("Parameters")

        # Header row is Excel row 2 (header=1).
        if need("Types"):
            headers = pd.read_excel(xls, sheet_name="Types", header=1, nrows=0).columns
            _check_headers(missing_headers, "Types", headers,
                            (Types.activity_type, Types.sectors, Types.energy_labels, Types.energy_price_init))
        else:
            missing_sheets.append("Types")

        # 3-row header at Excel rows 2-4 (skiprows=1, header=[0,1,2]).
        if need("Agents"):
            cols = pd.read_excel(xls, sheet_name="Agents", skiprows=1, header=[0, 1, 2], nrows=0).columns
            headers = flatten_header(cols)
            _check_headers(missing_headers, "Agents", headers,
                            required=(Agents.rates, Agents.profiles), prefixes=(Agents.types,))
        else:
            missing_sheets.append("Agents")

        # 2-row header at Excel rows 7-8 (skiprows=6, header=[0,1]).
        if need("Activities"):
            cols = pd.read_excel(xls, sheet_name="Activities", skiprows=6, header=[0, 1], nrows=0,
                                  keep_default_na=False).columns
            headers = flatten_header(cols)
            _check_headers(missing_headers, "Activities", headers,
                            required=(Activities.activities_names, Activities.activity_resolution,
                                      Activities.activity_type_act, Activities.activity_label,
                                      Activities.activity_agent),
                            prefixes=(Activities.periods_start,))
        else:
            missing_sheets.append("Activities")

        # Header row is Excel row 3 (header=2, no skiprows).
        if need("HourlyProfiles"):
            headers = pd.read_excel(xls, sheet_name="HourlyProfiles", header=2, nrows=0).columns.tolist()
            _check_headers(missing_headers, "HourlyProfiles", headers,
                            (HourlyProfiles.hour, HourlyProfiles.day, HourlyProfiles.month))
        else:
            missing_sheets.append("HourlyProfiles")

        # 2-row header at Excel rows 2-3 (header=[1,2]).
        if need("PriceProfiles"):
            cols = pd.read_excel(xls, sheet_name="PriceProfiles", header=[1, 2], nrows=0).columns
            headers = flatten_header(cols)
            _check_headers(missing_headers, "PriceProfiles", headers, prefixes=(PriceProfiles.interconnector,))
        else:
            missing_sheets.append("PriceProfiles")

        # 3-row header at Excel rows 2-4 (header=[1,2,3], no skiprows).
        # investment/fixed_om/variable_om are read as one column per period
        # (tech_year_cols in the real reader), so the flattened header text
        # is "<name> / <year>" - check by prefix, not exact match.
        if need("Technologies"):
            cols = pd.read_excel(xls, sheet_name="Technologies", header=[1, 2, 3], nrows=0).columns
            headers = flatten_header(cols)
            _check_headers(missing_headers, "Technologies", headers, required=(
                Technologies.tech_id, Technologies.category, Technologies.sector, Technologies.subsector,
                Technologies.main_activity, Technologies.name, Technologies.unit,
                Technologies.ec_lifetime, Technologies.cap2act,
                Technologies.dispatch_type, Technologies.hourly_profile, Technologies.social_perception,
                Technologies.perceived_complexity, Technologies.subsidy_subject, Technologies.feedin_subject,
                Technologies.shedding_capacity, Technologies.shedding_volume, Technologies.shedding_guarantee,
                Technologies.flexibility_form, Technologies.flexibility_activity, Technologies.flexibility_capacity,
                Technologies.flexibility_volume, Technologies.flexibility_range, Technologies.flexibility_losses,
                Technologies.flexibility_nonnegotiable, Technologies.buffer_up, Technologies.buffer_down,
                Technologies.buffer_capacity, Technologies.tech_stock_deploy, Technologies.tech_stock_exist,
            ), prefixes=(Technologies.investment, Technologies.fixed_om, Technologies.variable_om))
        else:
            missing_sheets.append("Technologies")

        # 3-row header at Excel rows 2-4 (header=[1,2,3], no skiprows). Only
        # fields the real reader actually looks up by name (planned
        # decommissioning / min / max stock blocks are positional-only, no
        # header text, same as Technologies' equivalent trailing blocks).
        if need("Infrastructure"):
            cols = pd.read_excel(xls, sheet_name="Infrastructure", header=[1, 2, 3], nrows=0).columns
            headers = flatten_header(cols)
            _check_headers(missing_headers, "Infrastructure", headers, required=(
                Infrastructure.tech_id, Infrastructure.category, Infrastructure.name, Infrastructure.unit,
                Infrastructure.ec_lifetime, Infrastructure.cap2act, Infrastructure.activity,
                Infrastructure.tech_stock_exist,
            ), prefixes=(Infrastructure.investment, Infrastructure.fixed_om))
        else:
            missing_sheets.append("Infrastructure")

        # Header row is Excel row 3 (header=2, no skiprows). No named header
        # constants exist for this sheet - the real reader cross-references
        # Activities' names instead of fixed header text, so this is just a
        # sheet/shape presence check.
        if need("EnergyBalance"):
            headers = pd.read_excel(xls, sheet_name="EnergyBalance", header=2, nrows=0).columns.tolist()
            if not headers:
                missing_headers.append("EnergyBalance: header row (row 3) is empty")
        else:
            missing_sheets.append("EnergyBalance")

        # Header row is Excel row 3 (skiprows=2).
        if need("Retrofitting"):
            headers = pd.read_excel(xls, sheet_name="Retrofitting", skiprows=2, nrows=0).columns.tolist()
            _check_headers(missing_headers, "Retrofitting", headers, (
                Retrofitting.tech_id_original, Retrofitting.tech_id_new,
                Retrofitting.enabled, Retrofitting.investment_cost,
            ))
        else:
            missing_sheets.append("Retrofitting")

        # No header at all (read positionally) - the real reader locates
        # three stacked blocks (Taxes/Feedins/Subsidies) via the literal
        # marker text "Units" in column B.
        if need("Policies"):
            policies_data = pd.read_excel(xls, sheet_name="Policies", header=None)
            unit_count = int((policies_data.iloc[:, 1] == "Units").sum()) if policies_data.shape[1] > 1 else 0
            if unit_count < 3:
                missing_headers.append("Policies: 'Units' marker rows (expected 3: Taxes/Feedins/Subsidies blocks)")
        else:
            missing_sheets.append("Policies")
    finally:
        xls.close()

    return {
        "compatible": not missing_sheets and not missing_headers,
        "missingSheets": missing_sheets,
        "missingHeaders": missing_headers,
    }


def check_sim_duckdb_compatibility(path: str) -> dict:
    try:
        con = duckdb.connect(path, read_only=True)
    except Exception as e:  # noqa: BLE001 - not a real DuckDB file at all; report, don't crash the endpoint
        return {"compatible": False, "missing": [f"<could not open as a DuckDB database: {e}>"]}
    try:
        existing = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    finally:
        con.close()
    missing = [t for t in _SIM_DUCKDB_TABLES if t not in existing]
    return {"compatible": not missing, "missing": missing}


def check_sim_compatibility(path: str) -> dict:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        return {"kind": "excel", **check_sim_excel_compatibility(path)}
    if ext in (".duckdb", ".db"):
        return {"kind": "duckdb", **check_sim_duckdb_compatibility(path)}
    raise ValueError(f"Unsupported file type for compatibility check: {ext!r}")
