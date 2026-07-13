import pytest
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from wrtds_py import WRTDS

@pytest.fixture(scope="session")
def choptank_raw_data():
    """
    Fixture to load the raw Choptank eList data from EGRET.
    Scope is 'session' so R only has to load this once per test run.
    """
    importr('EGRET')
    robjects.r('data(Choptank_eList, package="EGRET")')
    r_data = robjects.r('Choptank_eList')

    conversion_rules = robjects.default_converter + pandas2ri.converter
    with conversion_rules.context():
        # Convert R dataframes to Pandas
        daily_df = robjects.conversion.get_conversion().rpy2py(r_data.rx2('Daily'))
        sample_df = robjects.conversion.get_conversion().rpy2py(r_data.rx2('Sample'))
    
    # Convert R numeric dates to Pandas DatetimeIndex
    epoch = pd.Timestamp('1970-01-01')
    
    daily_df['datetime'] = pd.to_datetime(daily_df['Date'], unit='D', origin=epoch)
    daily_df = daily_df.set_index('datetime')
    
    sample_df['datetime'] = pd.to_datetime(sample_df['Date'], unit='D', origin=epoch)
    sample_df = sample_df.set_index('datetime')
    
    # Ensure Uncen is treated as an integer flag
    sample_df['Uncen'] = sample_df['Uncen'].astype(int)
    
    return sample_df, daily_df

def test_python_matches_or_beats_r(choptank_raw_data):
    """
    Tests that the Python WRTDS implementation achieves an RMSE that is
    less than or equal to the original R implementation.
    """
    df_sample, df_daily = choptank_raw_data
    
    # Fit the Python Model
    model = WRTDS()
    model.fit(df_sample, df_daily)
    
    # Check if the model was fitted successfully
    assert model._is_fitted, "Model _is_fitted flag was not set to True after calling .fit()"

    # Predict using the Python Model
    y_hat_sample = model.predict(df_sample)

    # Calculate RMSE for R
    res_r = np.abs(df_sample['ConcHat'] - df_sample['ConcAve'])
    rmse_r = np.sqrt(np.mean(res_r**2))

    # Calculate RMSE for Python
    res_py = np.abs(y_hat_sample['Conc'] - df_sample['ConcAve'])
    rmse_py = np.sqrt(np.mean(res_py**2))

    # Debugging output -> `uv run pytest -s`
    print("\n--- Results ---")
    print(f"RMSE of R predictions: {rmse_r:.3f} mg/L")
    print(f"RMSE of Python predictions: {rmse_py:.3f} mg/L")
    print("---------------")

    assert rmse_py <= rmse_r + 1e-6, f"Python RMSE ({rmse_py:.3f}) is worse than R RMSE ({rmse_r:.3f})"
