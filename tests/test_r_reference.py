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

def test_classic_wrtds_rmse(choptank_ground_truth, fitted_base_model):
    """
    Tests that the Python classic WRTDS implementation achieves an RMSE 
    less than or equal to the original R implementation.
    """
    df_wq, _ = choptank_ground_truth
    
    # Predict using the pre-fitted model
    y_hat_wq = fitted_base_model.predict(df_wq)

    # Extract arrays
    obs = df_wq['ConcAve'].values
    sim_r = df_wq['ConcHat'].values
    sim_py = y_hat_wq['ConcHat'].values
    num_nans = np.sum(np.isnan(sim_py))
    print(f"\nNumber of NaNs in Python predictions: {num_nans} out of {len(sim_py)}")

    # Calculate metrics
    rmse_r, r2_r = calculate_obs_metrics(obs, sim_r)
    rmse_py, r2_py = calculate_obs_metrics(obs, sim_py)

    print("\n--- Classic WRTDS Metrics ---")
    print(f"[Obs vs Sim] R      -> RMSE: {rmse_r:.3f} | R^2: {r2_r:.3f}")
    print(f"[Obs vs Sim] Python -> RMSE: {rmse_py:.3f} | R^2: {r2_py:.3f}")
    print("-----------------------------")

    assert rmse_py <= rmse_r + 1e-4, f"Python RMSE ({rmse_py:.6f}) is worse than R ({rmse_r:.6f})"
    assert r2_py >= r2_r - 1e-4, f"Python R^2 ({r2_py:.6f}) is worse than R ({r2_r:.6f})"

def test_kalman_filter_matches_r(choptank_ground_truth, fitted_base_model):
    """
    Tests that the Python WRTDS-K implementation produces Generated Concentrations
    (GenConc) that highly correlate with the R implementation.
    """
    df_wq, df_q = choptank_ground_truth
    
    # Run Kalman Filter using the pre-fitted base model
    k_model = WRTDSK(fitted_base_model, rho=0.9, niter=100, seed=42)
    df_q_kalman = k_model.filter(df_wq, df_q, verbose=False)
    
    # Extract arrays
    dates_obs = df_wq.index
    obs = df_wq['ConcAve'].values
    sim_r = df_q['GenConc'][dates_obs].values
    sim_py = df_q_kalman['ConcKalman'][dates_obs].values
    
    gen_r = df_q['GenConc'].values
    gen_py = df_q_kalman['ConcKalman'].values
    
    # Calculate metrics
    rmse_r, r2_r = calculate_obs_metrics(obs, sim_r)
    rmse_py, r2_py = calculate_obs_metrics(obs, sim_py)
    corr, mae = calculate_sim_metrics(gen_r, gen_py)

    mean_val = np.mean(gen_r)
    
    print("\n--- Kalman WRTDS Metrics ---")
    print(f"[Obs vs Sim] R      -> RMSE: {rmse_r:.3f} | R^2: {r2_r:.3f}")
    print(f"[Obs vs Sim] Python -> RMSE: {rmse_py:.3f} | R^2: {r2_py:.3f}")
    print(f"[R vs Python]       -> Corr: {corr:.3f} | MAE: {mae:.3f}")
    print("-----------------------------")
    
    assert corr > 0.90, f"Kalman correlation too low: {corr:.4f}"
    
    relative_error = mae / mean_val
    assert relative_error < 0.10, f"Kalman relative error too high: {relative_error:.2%}"
