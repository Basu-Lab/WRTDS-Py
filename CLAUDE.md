# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

*Last modified: 2026-11-17*

## Project

Python port of the USGS `EGRET` R package's **WRTDS** (Weighted Regressions on Time, Discharge, and Season) model and its **WRTDS-Kalman** AR(1) residual smoother. Correctness is defined by agreement with EGRET (v3.0.11) on the Choptank River reference dataset. `docs/methodology.md` documents the statistics (design matrix, Tobit likelihood, tricube weights, bias correction, AR(1) process) and should be consulted before changing numerical code.

## Commands

Managed with `uv` (Python ≥3.12, see `.python-version`).

```bash
uv sync                                            # install package + dev deps
uv run pytest                                      # full suite
uv run pytest tests/test_components.py             # fast unit tests
uv run pytest tests/test_models.py                 # synthetic-data smoke tests
uv run pytest tests/test_r_reference.py -s         # EGRET comparison; -s shows printed metrics
uv run pytest tests/test_components.py::test_tricube_weight   # single test
```

No linter/formatter is configured.

`tests/test_r_reference.py` requires a working R installation with the `EGRET` package installed (it is called through `rpy2`). It runs EGRET's `modelEstimation` and `WRTDSKalman` live on the Choptank example (~30 s in total):
- The session fixture `egret` returns an `EgretReference`: the Python inputs (`Q`, `LogQ`, `DecYear`, `Conc`, `Uncen`) kept apart from R's estimates (`ConcHat`, `ConcDay`, `GenConc`), so no R output reaches the Python models. EGRET's `DecYear` is passed through because it differs from what `prepare_data` computes by up to ~0.8 days.
- Metrics take date-indexed Series and compare R and Python on the same rows (`_common_rows`).
- Assertions are tolerance-based, not exact: no NaN Python estimates, Python RMSE/R² no worse than R, D_fluxbias within 1 percentage point, WRTDS-K (niter = 200 on both sides) correlation > 0.98, relative MAE < 4%, total flux within 1% of R, and mean bias within 3% on days more than 20 days from a sample; D_ratio no further from 1 than R's + 0.05. WRTDS-K is compared on the daily series, since on sample days it reproduces the observations by construction.

## Architecture

Public API is just `WRTDS` and `WRTDSK` (`src/wrtds_py/__init__.py`), plus `__version__` read from the installed package metadata.

**Data flow / column conventions.** `data.prepare_data` normalizes inputs and is the single place that defines the expected column names, mirroring EGRET's `Daily`/`Sample` frames:
- `df_q` (daily discharge): needs `Q` or `LogQ`; gets `DecYear`, `LogQ`. If any day has `Q = 0`, `LogQ` is computed from `Q + 0.001·mean(Q)` for every day (EGRET's zero-flow convention, with a `UserWarning`); `Q` itself is not modified.
- `df_wq` (sparse samples): needs `Conc` and `Uncen` (1 = uncensored, 0 = left-censored at `Conc`); gets `LogConc`, `DecYear`, and `LogQ` mapped from `df_q` by date (error if any sample date lacks discharge).
- A `DatetimeIndex` is required, or a `date`/`datetime`/`time`/`timestamp` column is promoted to the index.
- Output columns: `yHat`, `SE`, `ConcHat` (from `WRTDS.predict`) and `ConcKalman` (from `WRTDSK.filter`).

**Classic WRTDS (`models/classic.py`).** `fit` builds a fixed grid — 14 evenly spaced LogQ nodes × time nodes every 1/16 year — and calls `estimation.estimate_surface_node` independently for each node. Results go into `surface_` with shape `(n_logq, n_time, 3)` holding `[yHat, SE, ConcHat]`; the windows actually used per node go in `effective_windows_`. `predict` bilinearly interpolates the surface (`RegularGridInterpolator`), clipping inputs to the grid range so it never extrapolates. It returns a copy of the input with its original row order and index (`prepare_data` sorts by date, so rows are tagged with their position and the predictions written back in place); existing `yHat`/`SE`/`ConcHat` columns are overwritten. Fitted attributes use a trailing underscore (sklearn style); `BaseWRTDS` (`models/base.py`) only holds hyperparameters and these attributes.

**Per-node estimation (`estimation.py`).** For each target (logQ, time):
1. Edge adjustment widens the time window near the ends of the sample record.
2. Tricube weights (`weighting.py`) for time, discharge, and season (season uses a cyclic distance); the combined weight is their product.
3. If too few observations/uncensored observations, windows grow ×1.1 per step (caps: 15 yr time, 5 log-units discharge, 0.5 season; up to 50 steps). `WindowParams` is frozen; `expand()` returns a new instance.
4. A weighted Tobit (left-censored) MLE over the 5-term design matrix `[1, t−t0, logQ−logQ0, sin2πt, cos2πt]`, initialized from WLS, solved with L-BFGS-B. `_tobit_neg_log_likelihood` returns the objective *and* its analytic gradient (`jac=True`), with the censored term computed via `log_ndtr`. Don't fall back to finite differences: without censoring the WLS start is already the optimum, and L-BFGS-B then ends with ABNORMAL, which previously left ~9% of the nodes as NaN. `test_tobit_gradient` checks the gradient against finite differences.
5. `ConcHat = exp(yHat + σ²/2)` (log-normal bias correction).
Any failure returns NaNs for that node instead of raising; since each NaN node affects the four surrounding cells of the interpolation, NaNs in the predictions usually point to failed nodes in `surface_`.

**WRTDS-K (`models/kalman.py`).** Wraps an already-fitted `WRTDS` and follows EGRET's `WRTDSKalman` (see the WRTDS-K section of `docs/methodology.md`). Residuals are standardized, `z = (LogConc − yHat)/SE`, and treated as a unit-variance AR(1) with daily correlation `rho` (must be in [0, 1)); gaps are measured in whole days. Each of `niter` Monte Carlo iterations draws censored `z` from a truncated normal, picks one sample at random on days with several, and fills each gap between samples with an exact AR(1) bridge: an unconditional path from `scipy.signal.lfilter`, conditioned on both ends in closed form (vectorized, no per-day loop). Days before the first or after the last sample get independent N(0, 1), i.e. plain WRTDS. Concentration is `exp(yHat + z·SE)` with each day's SE and no `σ²/2` term (the Monte Carlo average is the bias correction); `ConcKalman` is the mean over iterations, so it equals the observation on uncensored sample days and tends to `ConcHat` mid-gap. `seed` (int, `np.random.Generator` or None) feeds a local `np.random.default_rng`; the global NumPy random state is never touched. Default `niter` is 200, as in EGRET.

`flux.calc_flux` converts concentration × discharge to kg/day (`cms` or `cfs`); it is not yet wired into the models.

## Notes

- The version lives only in `pyproject.toml`. To release: bump it, update `CHANGELOG.md` and `CITATION.cff`, tag `vX.Y.Z`, and publish a GitHub release; `.github/workflows/publish.yml` checks the tag matches the version, runs the non-R tests, builds with `uv build`, and uploads to PyPI via Trusted Publishing.
- `notebooks/` is gitignored; `docs/example.ipynb` is the tracked usage example.
