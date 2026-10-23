"""Per-period activity balances (IESA-Opt-style efficiency learning) for IESA-Sim.

IESA-Sim keeps ONE technology x activity coefficient matrix
(technologies.balancers.activity_balances). IESA-Opt additionally carries a
per-period fraction (activity_efficiency_improvement) that scales a coefficient to
coefficient * (1 - value) in that period. This module is the whole of Sim's support:

  - apply_period_balances(): at the start of each period, overwrite the matrix IN
    PLACE with that period's version. The technology objects hold row views of the
    same array (entities.py), so every invest/dispatch/post-processing step of the
    period sees it without further changes.
  - balances_for_period(): the same matrix as a copy, for the results that aggregate
    across all periods after the loop (emissions, policy cashflows, sector balance).
  - replay_per_period(): runs such an aggregation once per period with that period's
    matrix and keeps only that period's column of the named result fields.

Everything is a no-op when the database has no efficiency-improvement entries
(activity_eff_improvement is None): the legacy single-matrix path is untouched.
"""
import numpy as np


def has_period_balances(technologies):
    return getattr(technologies.balancers, 'activity_eff_improvement', None) is not None


def balances_for_period(technologies, iP):
    b = technologies.balancers
    if not has_period_balances(technologies):
        return b.activity_balances.copy()
    return b.activity_balances_base * (1.0 - b.activity_eff_improvement[:, :, iP])


def apply_period_balances(technologies, iP):
    """In-place update, so row views held elsewhere stay valid."""
    if not has_period_balances(technologies):
        return
    technologies.balancers.activity_balances[...] = balances_for_period(technologies, iP)


def replay_per_period(technologies, nP, fn, results, fields):
    """Call fn() once per period with that period's matrix; keep column iP of each
    results.<field> (periods on the LAST axis). Restores the base matrix afterwards."""
    if not has_period_balances(technologies):
        return fn()
    merged = {}
    for iP in range(nP):
        apply_period_balances(technologies, iP)
        fn()
        for f in fields:
            arr = np.array(getattr(results, f), dtype=float, copy=True)
            if iP == 0:
                merged[f] = np.array(arr, copy=True)
            merged[f][..., iP] = arr[..., iP]
    technologies.balancers.activity_balances[...] = technologies.balancers.activity_balances_base
    for f in fields:
        old = getattr(results, f)
        setattr(results, f, merged[f].reshape(np.shape(old)))
    return results
