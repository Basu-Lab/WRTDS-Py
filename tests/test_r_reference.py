import pytest
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

@pytest.fixture(scope="session") # Cache the R ground truth for the entire session
def r_egret_ground_truth():
    # Load R packages
    egret = importr('EGRET')
    
    # 1. Run the R models using raw R code
    robjects.r('data(Choptank_eList, package="EGRET")')
    robjects.r('eList_base <- EGRET::modelEstimation(Choptank_eList, verbose=FALSE)')
    robjects.r('eList_k <- EGRET::WRTDSKalman(eList_base, rho=0.9, niter=100, seed=42)')
    
    # Pull the R DataFrames
    r_base_daily = robjects.r('eList_base$Daily')
    r_kalman_daily = robjects.r('eList_k$Daily')
    
    # 2. rpy2 Conversion Context
    # We combine the default converter with the pandas converter and open a local context block.
    conversion_rules = robjects.default_converter + pandas2ri.converter
    with conversion_rules.context():
        df_base = robjects.conversion.get_conversion().rpy2py(r_base_daily)
        df_kalman = robjects.conversion.get_conversion().rpy2py(r_kalman_daily)
    
    return {
        "base": df_base,
        "kalman": df_kalman
    }

# TEST 1: Standard WRTDS
def test_standard_wrtds_output_structure(r_egret_ground_truth):
    df = r_egret_ground_truth["base"]
    
    assert "yHat" in df.columns, "R WRTDS did not yield yHat"
    assert "ConcDay" in df.columns, "R WRTDS did not yield ConcDay"
    assert "FluxDay" in df.columns, "R WRTDS did not yield FluxDay"
    
    assert (df["ConcDay"] >= 0).all(), "Found negative concentration values"
    assert (df["FluxDay"] >= 0).all(), "Found negative flux values"

# TEST 2: WRTDS-K
def test_kalman_wrtds_output_structure(r_egret_ground_truth):
    df_k = r_egret_ground_truth["kalman"]
    
    assert "GenConc" in df_k.columns, "R WRTDS-K did not yield GenConc"
    assert "GenFlux" in df_k.columns, "R WRTDS-K did not yield GenFlux"