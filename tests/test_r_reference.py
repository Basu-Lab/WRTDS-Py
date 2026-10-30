import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

from wrtds_py import WRTDS
from wrtds_py.models.kalman import WRTDSK


@pytest.fixture(scope="session")
def choptank_ground_truth():
    """
    Fixture to load the raw Choptank eList data and run both classic 
    WRTDS and WRTDS-K in R to serve as our ground truth.
    """
    importr('EGRET')
    
    # Load data and run R models
    robjects.r('data(Choptank_eList, package="EGRET")')
    robjects.r('eList_base <- EGRET::modelEstimation(Choptank_eList, verbose=FALSE)')
    robjects.r('eList_k <- EGRET::WRTDSKalman(eList_base, rho=0.9, niter=100, seed=42, verbose=FALSE)')
    
    # Extract DataFrames
    conversion_rules = robjects.default_converter + pandas2ri.converter
    with conversion_rules.context():
        r_sample_df = robjects.conversion.get_conversion().rpy2py(robjects.r('eList_base$Sample'))
        r_daily_k_df = robjects.conversion.get_conversion().rpy2py(robjects.r('eList_k$Daily'))
    
    # Data Preparation for Python
    epoch = pd.Timestamp('1970-01-01')
    
    df_q = r_daily_k_df.copy()
    df_q['datetime'] = pd.to_datetime(df_q['Date'], unit='D', origin=epoch)
    df_q = df_q.set_index('datetime')
    
    df_wq = r_sample_df.copy()
    df_wq['datetime'] = pd.to_datetime(df_wq['Date'], unit='D', origin=epoch)
    df_wq = df_wq.set_index('datetime')
    df_wq['Uncen'] = df_wq['Uncen'].astype(int)
    df_wq['Conc'] = df_wq['ConcHigh'].values
    
    return df_wq, df_q

@pytest.fixture(scope="session")
def fitted_base_model(choptank_ground_truth):
    """
    Fixture that fits the base Python WRTDS model exactly once per session.
    It depends on the choptank_ground_truth fixture for data.
    """
    df_wq, df_q = choptank_ground_truth
    
    model = WRTDS()
    model.fit(df_wq, df_q, verbose=False)
    
    return model

def calculate_obs_metrics(y_obs, y_sim):
    mask = ~np.isnan(y_obs) & ~np.isnan(y_sim)
    y_obs, y_sim = y_obs[mask], y_sim[mask]

    res = y_obs - y_sim
    rmse = np.sqrt(np.mean(res**2))
    ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
    ss_res = np.sum(res**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    return rmse, r2

def calculate_sim_metrics(y_r, y_py):
    mask = ~np.isnan(y_r) & ~np.isnan(y_py)
    y_r, y_py = y_r[mask], y_py[mask]
    corr = np.corrcoef(y_r, y_py)[0, 1]
    mae = np.mean(np.abs(y_r - y_py))
    return corr, mae

def calculate_diagnostics(c_obs, q_obs, c_base, full_c_kalman=None):
    """
    Helper to calculate D_fluxbias and D_ratio directly from arrays.
    Dates must be aligned before passing.
    """
    # 1. D_fluxbias: requires strictly aligned base estimates and observations
    mask = ~np.isnan(c_obs) & ~np.isnan(q_obs) & ~np.isnan(c_base)
    c_obs_m, q_obs_m, c_base_m = c_obs[mask], q_obs[mask], c_base[mask]

    L_sim = np.sum(c_base_m * q_obs_m)
    L_obs = np.sum(c_obs_m * q_obs_m)
    
    d_fluxbias = ((L_sim - L_obs) / L_sim) * 100.0 if L_sim != 0 else np.nan

    # 2. D_ratio: Ratio of Standard Deviations
    # If Kalman is not provided (Classic test), we use the base estimates for ratio 
    if full_c_kalman is None:
        full_c_kalman = c_base 

    std_obs = np.std(c_obs[~np.isnan(c_obs)], ddof=1)
    std_sim = np.std(full_c_kalman[~np.isnan(full_c_kalman)], ddof=1)
    
    d_ratio = (std_obs / std_sim) if std_sim != 0 else np.nan

    return d_fluxbias, d_ratio

def test_classic_wrtds_rmse(choptank_ground_truth, fitted_base_model):
    """
    Tests that the Python classic WRTDS implementation achieves an RMSE 
    less than or equal to the original R implementation, and verifies diagnostic metrics.
    """
    df_wq, df_q = choptank_ground_truth
    
    # Predict using the pre-fitted model
    y_hat_wq = fitted_base_model.predict(df_wq)

    # Extract arrays
    obs = df_wq['ConcAve'].values
    q_obs = df_wq['Q'].values
    sim_r = df_wq['ConcHat'].values
    sim_py = y_hat_wq['ConcHat'].values
    
    num_nans = np.sum(np.isnan(sim_py))
    print(f"\nNumber of NaNs in Python predictions: {num_nans} out of {len(sim_py)}")

    # Calculate metrics
    rmse_r, r2_r = calculate_obs_metrics(obs, sim_r)
    rmse_py, r2_py = calculate_obs_metrics(obs, sim_py)
    
    # Calculate Diagnostics
    df_q_base_py = fitted_base_model.predict(df_q)
    flux_r, rat_r = calculate_diagnostics(obs, q_obs, c_base=sim_r, full_c_kalman=df_q['ConcDay'].values)
    flux_py, rat_py = calculate_diagnostics(obs, q_obs, c_base=sim_py, full_c_kalman=df_q_base_py['ConcHat'].values)

    print("\n--- Classic WRTDS Metrics ---")
    print(f"[Obs vs Sim] R      -> RMSE: {rmse_r:.3f} | R^2: {r2_r:.3f}")
    print(f"[Obs vs Sim] Python -> RMSE: {rmse_py:.3f} | R^2: {r2_py:.3f}")
    print(f"[Diagnostic] R      -> D_fluxbias: {flux_r:.3f}% | D_ratio: {rat_r:.3f}")
    print(f"[Diagnostic] Python -> D_fluxbias: {flux_py:.3f}% | D_ratio: {rat_py:.3f}")
    print("-----------------------------")

    assert rmse_py <= rmse_r + 1e-4, f"Python RMSE ({rmse_py:.6f}) is worse than R ({rmse_r:.6f})"
    assert r2_py >= r2_r - 1e-4, f"Python R^2 ({r2_py:.6f}) is worse than R ({r2_r:.6f})"
    assert abs(flux_py - flux_r) < 1.0, f"Python D_fluxbias diverges from R (Py: {flux_py:.2f}%, R: {flux_r:.2f}%)"

def test_kalman_filter_matches_r(choptank_ground_truth, fitted_base_model):
    """
    Tests that the Python WRTDS-K implementation produces Generated Concentrations
    (GenConc) that highly correlate with the R implementation, and verifies diagnostics.
    """
    df_wq, df_q = choptank_ground_truth
    
    # Run Kalman Filter using the pre-fitted base model
    k_model = WRTDSK(fitted_base_model, rho=0.9, niter=100, seed=42)
    df_q_kalman = k_model.filter(df_wq, df_q, verbose=False)
    
    # Extract arrays
    dates_obs = df_wq.index
    obs = df_wq['ConcAve'].values
    q_obs = df_wq['Q'].values
    sim_r_base = df_wq['ConcHat'].values
    sim_r_kalman = df_q['GenConc'][dates_obs].values
    
    sim_py_base = fitted_base_model.predict(df_wq)['ConcHat'].values
    sim_py_kalman = df_q_kalman['ConcKalman'][dates_obs].values
    
    gen_r = df_q['GenConc'].values
    gen_py = df_q_kalman['ConcKalman'].values
    
    # Calculate metrics
    rmse_r, r2_r = calculate_obs_metrics(obs, sim_r_kalman)
    rmse_py, r2_py = calculate_obs_metrics(obs, sim_py_kalman)
    corr, mae = calculate_sim_metrics(gen_r, gen_py)
    
    # Calculate Diagnostics (Using Kalman full series for D_ratio)
    flux_r, rat_r = calculate_diagnostics(obs, q_obs, c_base=sim_r_base, full_c_kalman=gen_r)
    flux_py, rat_py = calculate_diagnostics(obs, q_obs, c_base=sim_py_kalman, full_c_kalman=gen_py)

    mean_val = np.mean(gen_r)
    
    print("\n--- Kalman WRTDS Metrics ---")
    print(f"[Obs vs Sim] R      -> RMSE: {rmse_r:.3f} | R^2: {r2_r:.3f}")
    print(f"[Obs vs Sim] Python -> RMSE: {rmse_py:.3f} | R^2: {r2_py:.3f}")
    print(f"[R vs Python]       -> Corr: {corr:.3f} | MAE: {mae:.3f}")
    print(f"[Diagnostic] R      -> D_fluxbias: {flux_r:.3f}% | D_ratio: {rat_r:.3f}")
    print(f"[Diagnostic] Python -> D_fluxbias: {flux_py:.3f}% | D_ratio: {rat_py:.3f}")
    print("-----------------------------")
    
    assert corr > 0.90, f"Kalman correlation too low: {corr:.4f}"
    
    relative_error = mae / mean_val
    assert relative_error < 0.10, f"Kalman relative error too high: {relative_error:.2%}"
    
    # Asserting that Python D_ratio behaves similarly to the R reference
    # or outperforms it by being closer to the ideal ratio of 1.0
    dist_py = abs(rat_py - 1.0)
    dist_r = abs(rat_r - 1.0)
    
    assert dist_py <= dist_r + 0.05, (
        f"Python D_ratio is worse (further from 1.0) than R "
        f"(Py: {rat_py:.3f}, R: {rat_r:.3f})"
    )
