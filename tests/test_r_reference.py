import pytest
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

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

def test_classic_wrtds_rmse(choptank_ground_truth, fitted_base_model):
    """
    Tests that the Python classic WRTDS implementation achieves an RMSE 
    less than or equal to the original R implementation.
    """
    df_wq, _ = choptank_ground_truth
    
    # Predict using the pre-fitted model
    y_hat_wq = fitted_base_model.predict(df_wq)

    # R RMSE
    res_r = np.abs(df_wq['ConcHat'] - df_wq['ConcAve'])
    rmse_r = np.sqrt(np.mean(res_r**2))

    # Python RMSE
    res_py = np.abs(y_hat_wq['ConcHat'] - df_wq['ConcAve'])
    rmse_py = np.sqrt(np.mean(res_py**2))

    assert rmse_py <= rmse_r + 1e-6, f"Python RMSE ({rmse_py:.6f}) is worse than R RMSE ({rmse_r:.6f})"

def test_kalman_filter_matches_r(choptank_ground_truth, fitted_base_model):
    """
    Tests that the Python WRTDS-K implementation produces Generated Concentrations
    (GenConc) that highly correlate with the R implementation.
    """
    df_wq, df_q = choptank_ground_truth
    
    # Run Kalman Filter using the pre-fitted base model
    k_model = WRTDSK(fitted_base_model, rho=0.9, niter=100, seed=42)
    df_q_kalman = k_model.filter(df_wq, df_q, verbose=False)
    
    # Compare GenConc
    r_gen_conc = df_q['GenConc'].values
    py_gen_conc = df_q_kalman['GenConc'].values
    
    correlation = np.corrcoef(r_gen_conc, py_gen_conc)[0, 1]
    mae = np.mean(np.abs(r_gen_conc - py_gen_conc))
    mean_val = np.mean(r_gen_conc)
    
    print(f"\n--- Kalman Debug Info ---")
    print(f"Pearson Correlation: {correlation:.6f}")
    print(f"Mean Absolute Error: {mae:.6f} (Mean Conc: {mean_val:.6f})")
    print(f"-------------------------")
    
    assert correlation > 0.90, f"Kalman correlation too low: {correlation:.4f}"
    
    relative_error = mae / mean_val
    assert relative_error < 0.10, f"Kalman relative error too high: {relative_error:.2%}"
