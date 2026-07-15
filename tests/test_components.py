import pytest
import numpy as np
import pandas as pd

from wrtds_py import WRTDS
from wrtds_py.base import WindowParams
from wrtds_py.models.utils import tricube_weight, cyclic_distance
from wrtds_py.models._estimation import estimate_surface_node, NodeConfig

@pytest.fixture(scope="session")
def station_data():
    """
    Loads the static CSV test data for the specific station.
    Make sure tests/data/station_wq.csv and station_q.csv exist!
    """
    # Load the test data
    df = pd.read_csv("tests/data.csv", index_col='Date', parse_dates=True)

    # Separate into 'Q'
    df_q = df[['Q']].copy()

    # And 'WQ'
    df_wq = df.dropna(subset=['Conc', 'Uncen'])
    df_wq['Q'] = np.nan
    for date in df_wq.index:
        df_wq.loc[date, 'Q'] = df_q.loc[date, 'Q']

    # Re-cast the censorship flag to integer
    df_wq['Uncen'] = df_wq['Uncen'].astype(int)
    
    return df_wq, df_q

# --- UNIT TESTS FOR UTILITIES ---

def test_tricube_weight():
    """Tests the tricube distance weighting mathematical function."""
    half_window = 1.0
    
    # Distance of 0 should have maximum weight (1.0)
    assert tricube_weight(0.0, half_window) == 1.0
    
    # Distance exactly at the window edge should be 0.0
    assert tricube_weight(1.0, half_window) == 0.0
    
    # Distance outside the window should be strictly 0.0
    assert tricube_weight(1.5, half_window) == 0.0
    
    # Test array broadcasting
    dists = np.array([0.0, 0.5, 1.0])
    weights = tricube_weight(dists, half_window)
    assert np.allclose(weights, [1.0, 0.66992, 0.0])

def test_cyclic_distance():
    """Tests that seasonal wrapping calculates distance across the new year."""
    target_time = 0.1  # e.g., early February
    sample_time = np.array([0.9])  # e.g., late November of a previous year
    
    # The linear distance is 0.8, but the cyclic (seasonal) distance should be 0.2
    dist = cyclic_distance(target_time, sample_time)
    assert np.isclose(dist[0], 0.2)

def test_window_params_immutability():
    """Tests that the frozen dataclass properly returns a new instance on expansion."""
    initial = WindowParams(time=7.0, discharge=2.0, season=0.5)
    expanded = initial.expand(factor=1.1)
    
    # The new instance should be expanded
    assert np.isclose(expanded.time, 7.7)
    
    # The original instance MUST remain unchanged
    assert initial.time == 7.0
    assert initial != expanded

# --- UNIT TESTS FOR FITTING COMPONENTS ---

def test_data_validation(station_data):
    """Tests that the BaseWRTDS data parser creates required columns."""
    df_wq, df_q = station_data
    model = WRTDS()
    
    df_wq_clean, df_q_clean = model._validate_data(df_wq, df_q)
    
    # Check that DecYear was successfully mapped onto the indexes
    assert 'DecYear' in df_wq_clean.columns
    assert 'DecYear' in df_q_clean.columns
    
    # Check that LogQ was created and populated
    assert 'LogQ' in df_q_clean.columns
    assert 'LogQ' in df_wq_clean.columns
    
    # Check that Concentration arrays were split for Tobit
    assert 'ConcHigh' in df_wq_clean.columns
    assert 'ConcLow' in df_wq_clean.columns

def test_single_node_estimation(station_data):
    """
    Isolates and tests the Tobit regression for a single grid node, 
    verifying that the optimization routine converges and returns expected types.
    """
    df_wq, df_q = station_data
    model = WRTDS()
    
    # 1. Prep data using the model's internal validator
    df_wq_clean, _ = model._validate_data(df_wq, df_q)
    
    sample_data = {
        'time': df_wq_clean['DecYear'].values,
        'logq': df_wq_clean['LogQ'].values,
        'y_high': np.log(df_wq_clean['ConcHigh'].values),
        'uncen': df_wq_clean['Uncen'].values
    }
    
    # 2. Configure a specific node somewhere in the middle of our data
    target_time = df_wq_clean['DecYear'].median()
    target_logq = df_wq_clean['LogQ'].median()
    
    node = NodeConfig(
        target_logq=target_logq,
        target_time=target_time,
        min_obs=50,
        min_uncen=30,
        edge_adjust=True
    )
    
    initial_windows = WindowParams(time=7.0, discharge=2.0, season=0.5)
    
    # 3. Run the isolated estimation
    result = estimate_surface_node(node, sample_data, initial_windows)
    
    # 4. Assertions
    assert not np.isnan(result.y_hat), "y_hat returned NaN"
    assert result.sigma > 0, "Sigma should be strictly positive"
    assert result.conc_hat > 0, "Generated concentration should be strictly positive"
    assert isinstance(result.final_windows, WindowParams), "Did not return a WindowParams object"
