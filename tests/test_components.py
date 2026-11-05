import pytest
import numpy as np
import pandas as pd

from wrtds_py import WRTDS
from wrtds_py.weighting import tricube_weight, cyclic_distance
from scipy.optimize import check_grad

from wrtds_py.estimation import estimate_surface_node, NodeConfig, WindowParams, _tobit_neg_log_likelihood

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
    
    df_wq_clean, df_q_clean = model._prepare_data(df_wq, df_q)
    
    # Check that DecYear was successfully mapped onto the indexes
    assert 'DecYear' in df_wq_clean.columns
    assert 'DecYear' in df_q_clean.columns
    
    # Check that LogQ was created and populated
    assert 'LogQ' in df_q_clean.columns
    assert 'LogQ' in df_wq_clean.columns
    
    # Check that LogConc was created and populated
    assert 'LogConc' in df_wq_clean.columns

def test_zero_flow_shift(station_data):
    """Tests that zero-flow days get a finite LogQ from the EGRET shift (0.1% of mean Q)."""
    df_wq, df_q = station_data
    df_q = df_q.copy()
    df_wq = df_wq.copy()
    zero_date = df_wq.index[0]
    df_q.loc[zero_date, 'Q'] = 0.0
    df_wq.loc[zero_date, 'Q'] = 0.0

    with pytest.warns(UserWarning, match="zero flow"):
        df_wq_clean, df_q_clean = WRTDS()._prepare_data(df_wq, df_q)

    shift = 0.001 * df_q['Q'].mean()
    assert np.all(np.isfinite(df_q_clean['LogQ']))
    assert np.allclose(df_q_clean['LogQ'], np.log(df_q['Q'] + shift))
    assert df_wq_clean.loc[zero_date, 'LogQ'] == pytest.approx(np.log(shift))
    # Q itself is left unchanged
    assert df_q_clean.loc[zero_date, 'Q'] == 0.0

def test_single_node_estimation(station_data):
    """
    Isolates and tests the Tobit regression for a single grid node, 
    verifying that the optimization routine converges and returns expected types.
    """
    df_wq, df_q = station_data
    model = WRTDS()
    
    # 1. Prep data using the model's internal validator
    df_wq_clean, _ = model._prepare_data(df_wq, df_q)
    
    sample_data = {
        'time': df_wq_clean['DecYear'].values,
        'logq': df_wq_clean['LogQ'].values,
        'y': df_wq_clean['LogConc'].values,
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
        adjust_edges=True
    )
    
    initial_windows = WindowParams(time=7.0, discharge=2.0, season=0.5)
    
    # 3. Run the isolated estimation
    result = estimate_surface_node(node, sample_data, initial_windows)
    
    # 4. Assertions
    assert not np.isnan(result.y_hat), "y_hat returned NaN"
    assert result.sigma > 0, "Sigma should be strictly positive"
    assert result.conc_hat > 0, "Generated concentration should be strictly positive"
    assert isinstance(result.final_windows, WindowParams), "Did not return a WindowParams object"

@pytest.mark.parametrize("censored_fraction", [0.0, 0.3])
def test_tobit_gradient(censored_fraction):
    """
    Verifies the analytic gradient of the Tobit likelihood against finite differences,
    including censored samples far into the lower tail.
    """
    rng = np.random.default_rng(0)
    n = 60

    X = np.column_stack((np.ones(n), rng.normal(size=(n, 4))))
    model_data = {
        "X": X,
        "y": X @ rng.normal(size=5) + rng.normal(scale=0.5, size=n),
        "uncen": (rng.random(n) >= censored_fraction).astype(int),
        "weights": rng.uniform(0.1, 2.0, size=n),
    }

    for _ in range(5):
        x = np.append(rng.normal(size=5), rng.uniform(0.1, 1.0))
        func = lambda x: _tobit_neg_log_likelihood(x, model_data)[0]
        grad = lambda x: _tobit_neg_log_likelihood(x, model_data)[1]
        error = check_grad(func, grad, x) / max(1.0, np.linalg.norm(grad(x)))
        assert error < 1e-5, f"Analytic gradient deviates from finite differences (relative error {error:.2e})"
