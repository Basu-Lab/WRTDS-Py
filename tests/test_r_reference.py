"""
Validation against the USGS EGRET R package on its Choptank River example.

EGRET is run once per session through rpy2. The Python models only receive the input
columns they need, so none of R's estimates (yHat, SE, ConcHat, ...) reach them. EGRET's
DecYear and LogQ are passed on so that both implementations share the same time axis.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

from wrtds_py import WRTDS, WRTDSK

RHO, NITER, SEED = 0.9, 100, 42


@dataclass(frozen=True)
class EgretReference:
    df_wq: pd.DataFrame       # Python input: Q, LogQ, DecYear, Conc, Uncen
    df_q: pd.DataFrame        # Python input: Q, LogQ, DecYear
    obs: pd.Series            # Observed concentration (ConcAve) on sample days
    conc_sample: pd.Series    # R classic WRTDS (ConcHat) on sample days
    conc_daily: pd.Series     # R classic WRTDS (ConcDay) on every day
    conc_kalman: pd.Series    # R WRTDS-K (GenConc) on every day


def _r_to_pandas(expr: str) -> pd.DataFrame:
    with (robjects.default_converter + pandas2ri.converter).context():
        df = robjects.conversion.get_conversion().rpy2py(robjects.r(expr))
    # R Dates arrive as days since the Unix epoch
    return df.set_index(pd.to_datetime(df["Date"], unit="D", origin="unix"))


@pytest.fixture(scope="session")
def egret() -> EgretReference:
    importr("EGRET")
    robjects.r('data(Choptank_eList, package="EGRET")')
    robjects.r("eList <- EGRET::modelEstimation(Choptank_eList, verbose=FALSE)")
    robjects.r(f"eList <- EGRET::WRTDSKalman(eList, rho={RHO}, niter={NITER}, seed={SEED}, verbose=FALSE)")

    sample = _r_to_pandas("eList$Sample")
    daily = _r_to_pandas("eList$Daily")

    # Censored samples are fitted at their reporting limit (ConcHigh)
    df_wq = sample[["Q", "LogQ", "DecYear", "Uncen"]].assign(
        Conc=sample["ConcHigh"], Uncen=sample["Uncen"].astype(int)
    )

    return EgretReference(
        df_wq=df_wq,
        df_q=daily[["Q", "LogQ", "DecYear"]],
        obs=sample["ConcAve"],
        conc_sample=sample["ConcHat"],
        conc_daily=daily["ConcDay"],
        conc_kalman=daily["GenConc"],
    )


@pytest.fixture(scope="session")
def py_model(egret: EgretReference) -> WRTDS:
    return WRTDS().fit(egret.df_wq, egret.df_q, verbose=False)


@pytest.fixture(scope="session")
def py_kalman(egret: EgretReference, py_model: WRTDS) -> pd.Series:
    kalman = WRTDSK(py_model, rho=RHO, niter=NITER, seed=SEED)
    return kalman.filter(egret.df_wq, egret.df_q, verbose=False)["ConcKalman"]


# --------------------------------------------------------------------------
# Metrics. Inputs are Series indexed by date, so they align by label.
# --------------------------------------------------------------------------

def _common_rows(*series: pd.Series) -> list[pd.Series]:
    """Align the series by date and keep only the rows where all of them are finite."""
    df = pd.concat(series, axis=1, keys=range(len(series))).replace([np.inf, -np.inf], np.nan).dropna()
    return [df[k] for k in range(len(series))]


def rmse_r2(obs: pd.Series, sim: pd.Series) -> tuple[float, float]:
    res = obs - sim
    rmse = np.sqrt(np.mean(res**2))
    r2 = 1.0 - np.sum(res**2) / np.sum((obs - obs.mean())**2)
    return rmse, r2


def flux_bias(obs: pd.Series, sim: pd.Series, q: pd.Series) -> float:
    """D_fluxbias (%): relative difference between estimated and observed flux on sample days."""
    load_sim, load_obs = np.sum(sim * q), np.sum(obs * q)
    return 100.0 * (load_sim - load_obs) / load_sim


def std_ratio(obs: pd.Series, daily: pd.Series) -> float:
    """D_ratio: std. dev. of the observations over that of the daily estimates (ideal: 1)."""
    return obs.dropna().std(ddof=1) / daily.dropna().std(ddof=1)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_classic_wrtds_matches_r(egret: EgretReference, py_model: WRTDS):
    py_sample = py_model.predict(egret.df_wq)["ConcHat"]
    py_daily = py_model.predict(egret.df_q)["ConcHat"]

    # Compare both implementations on the same samples
    obs, q, r_sim, py_sim = _common_rows(egret.obs, egret.df_wq["Q"], egret.conc_sample, py_sample)

    rmse_r, r2_r = rmse_r2(obs, r_sim)
    rmse_py, r2_py = rmse_r2(obs, py_sim)
    flux_r, flux_py = flux_bias(obs, r_sim, q), flux_bias(obs, py_sim, q)
    ratio_r, ratio_py = std_ratio(egret.obs, egret.conc_daily), std_ratio(egret.obs, py_daily)

    print(f"\n--- Classic WRTDS ({len(obs)} of {len(egret.obs)} samples compared, "
          f"{py_sample.isna().sum()} Python NaNs) ---")
    print(f"R      -> RMSE: {rmse_r:.3f} | R^2: {r2_r:.3f} | D_fluxbias: {flux_r:.3f}% | D_ratio: {ratio_r:.3f}")
    print(f"Python -> RMSE: {rmse_py:.3f} | R^2: {r2_py:.3f} | D_fluxbias: {flux_py:.3f}% | D_ratio: {ratio_py:.3f}")

    assert rmse_py <= rmse_r + 1e-4, f"Python RMSE ({rmse_py:.6f}) is worse than R ({rmse_r:.6f})"
    assert r2_py >= r2_r - 1e-4, f"Python R^2 ({r2_py:.6f}) is worse than R ({r2_r:.6f})"
    assert abs(flux_py - flux_r) < 1.0, f"Python D_fluxbias diverges from R (Py: {flux_py:.2f}%, R: {flux_r:.2f}%)"


def test_kalman_filter_matches_r(egret: EgretReference, py_kalman: pd.Series):
    # On sample days WRTDS-K reproduces the observations by construction, so the
    # comparison is made on the full daily series instead
    r_daily, py_daily = _common_rows(egret.conc_kalman, py_kalman)

    corr = np.corrcoef(r_daily, py_daily)[0, 1]
    relative_mae = np.mean(np.abs(r_daily - py_daily)) / r_daily.mean()
    ratio_r, ratio_py = std_ratio(egret.obs, egret.conc_kalman), std_ratio(egret.obs, py_kalman)

    print(f"\n--- WRTDS-K ({len(r_daily)} days compared) ---")
    print(f"R vs Python -> Corr: {corr:.3f} | Relative MAE: {relative_mae:.2%}")
    print(f"D_ratio     -> R: {ratio_r:.3f} | Python: {ratio_py:.3f}")

    assert corr > 0.90, f"Kalman correlation too low: {corr:.4f}"
    assert relative_mae < 0.10, f"Kalman relative error too high: {relative_mae:.2%}"
    # Python's D_ratio may be closer to the ideal of 1.0 than R's, but not much further from it
    assert abs(ratio_py - 1.0) <= abs(ratio_r - 1.0) + 0.05, (
        f"Python D_ratio is further from 1.0 than R (Py: {ratio_py:.3f}, R: {ratio_r:.3f})"
    )
