import numpy as np
import pandas as pd
import pytest

from wrtds_py import WRTDS
from wrtds_py.models.kalman import WRTDSK

# --------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_data():
    """
    Builds a small but realistically-shaped dataset:
      - Q: log-normal, serially correlated (AR(1) in log-space) -> always > 0
      - Conc: exponential, loosely tied to flow -> always > 0
      - Uncen: mostly 1 (uncensored), a minority of 0 (left-censored)
    """
    rng = np.random.default_rng(42)

    num_days = 2 * 365
    dates = pd.date_range("2018-01-01", periods=num_days, freq="D")

    # --- Discharge: AR(1) on log-scale, so it's positive and autocorrelated ---
    phi = 0.99
    long_run_mean = np.log(5.0)
    sigma_ar = 0.15

    log_q = np.empty(num_days)
    log_q[0] = long_run_mean
    deviations = rng.normal(0.0, sigma_ar, size=num_days - 1)
    for t in range(1, num_days):
        log_q[t] = phi * log_q[t - 1] + (1 - phi) * long_run_mean + deviations[t - 1]

    q = np.exp(log_q)
    assert np.all(q > 0)

    df_q = pd.DataFrame({"Q": q}, index=dates)
    df_q.index.name = "Date"

    # --- Water quality samples: irregular subset of days ---
    num_samples = 80
    sample_idx = np.sort(rng.choice(num_days, size=num_samples, replace=False))
    sample_dates = dates[sample_idx]

    # Exponential concentration, mean loosely decreasing with flow (dilution-ish),
    # just to give the regression something structured to find.
    mean_conc = 1.0 / (0.3 + 0.05 * np.log1p(q[sample_idx]))
    conc = rng.exponential(scale=mean_conc)
    assert np.all(conc > 0)

    # Mostly uncensored (1s), ~10% censored (0s), matching real WQ data
    uncen = rng.choice([1, 0], size=num_samples, p=[0.9, 0.1])
    conc[uncen == 0] = 0.1 # detection limit

    df_wq = pd.DataFrame(
        {"Q": q[sample_idx], "Conc": conc, "Uncen": uncen},
        index=sample_dates,
    )
    df_wq.index.name = "Date"

    return df_wq, df_q


# --------------------------------------------------------------------------
# End-to-end tests for WRTDS and WRTDS-K
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_base_model(synthetic_data):
    df_wq, df_q = synthetic_data
    
    model = WRTDS()
    model.fit(df_wq, df_q, verbose=False)
    
    return model

def test_wrtds_fit_predict_smoke(fitted_base_model, synthetic_data):
    df_wq, _ = synthetic_data

    model = fitted_base_model

    preds = model.predict(df_wq)

    assert {"yHat", "SE", "ConcHat"}.issubset(preds.columns)
    assert not preds[["yHat", "SE", "ConcHat"]].isna().any().any()
    assert (preds["SE"] > 0).all()
    assert (preds["ConcHat"] > 0).all()
    assert len(preds) == len(df_wq)


def test_wrtdsk_filter_smoke(fitted_base_model, synthetic_data):
    df_wq, df_q = synthetic_data

    k_model = WRTDSK(fitted_base_model, rho=0.9, niter=5, seed=42)
    result = k_model.filter(df_wq, df_q, verbose=False)

    assert "ConcKalman" in result.columns
    assert not result["ConcKalman"].isna().any()
    assert (result["ConcKalman"] > 0).all()
    assert len(result) == len(df_q)


def test_wrtds_predict_preserves_row_order(fitted_base_model, synthetic_data):
    _, df_q = synthetic_data

    model = fitted_base_model

    shuffled = df_q.sample(frac=1, random_state=0)
    preds = model.predict(shuffled)
    expected = model.predict(df_q).loc[shuffled.index]

    assert preds.index.equals(shuffled.index)
    np.testing.assert_allclose(preds[["yHat", "SE", "ConcHat"]], expected[["yHat", "SE", "ConcHat"]])


def test_wrtdsk_rejects_invalid_rho(fitted_base_model):
    with pytest.raises(ValueError, match="rho"):
        WRTDSK(fitted_base_model, rho=1.0)


def test_wrtdsk_reproduces_uncensored_samples(fitted_base_model, synthetic_data):
    df_wq, df_q = synthetic_data

    result = WRTDSK(fitted_base_model, niter=5, seed=0).filter(df_wq, df_q, verbose=False)

    uncensored = df_wq[df_wq["Uncen"] == 1]
    np.testing.assert_allclose(result.loc[uncensored.index, "ConcKalman"], uncensored["Conc"], rtol=1e-10)


def test_wrtdsk_draws_a_new_path_each_iteration(fitted_base_model, synthetic_data):
    # Without censoring the gaps must still be random draws, not a fixed interpolation
    df_wq, df_q = synthetic_data
    uncensored = df_wq[df_wq["Uncen"] == 1]

    a = WRTDSK(fitted_base_model, niter=1, seed=1).filter(uncensored, df_q, verbose=False)
    b = WRTDSK(fitted_base_model, niter=1, seed=2).filter(uncensored, df_q, verbose=False)

    gap_days = df_q.index.difference(uncensored.index)
    assert not np.allclose(a.loc[gap_days, "ConcKalman"], b.loc[gap_days, "ConcKalman"])


def test_wrtdsk_returns_to_wrtds_in_long_gaps(fitted_base_model, synthetic_data):
    # Remove all samples from a 300-day window: far from both ends of the gap the
    # AR(1) memory has decayed, so WRTDS-K should agree with classic WRTDS on average
    df_wq, df_q = synthetic_data
    gap_start, gap_end = pd.Timestamp("2018-06-01"), pd.Timestamp("2019-03-28")
    outside_gap = (df_wq.index < gap_start) | (df_wq.index > gap_end)

    result = WRTDSK(fitted_base_model, niter=500, seed=0).filter(df_wq[outside_gap], df_q, verbose=False)

    mid_gap = result.loc[gap_start + pd.Timedelta(days=100) : gap_end - pd.Timedelta(days=100)]
    ratio = mid_gap["ConcKalman"].mean() / mid_gap["ConcHat"].mean()
    assert abs(ratio - 1.0) < 0.05, f"Mid-gap WRTDS-K / WRTDS ratio: {ratio:.3f}"


def test_wrtdsk_seeding_is_local_and_reproducible(fitted_base_model, synthetic_data):
    df_wq, df_q = synthetic_data

    np.random.seed(123)
    expected_global = np.random.random()
    np.random.seed(123)

    a = WRTDSK(fitted_base_model, niter=3, seed=7).filter(df_wq, df_q, verbose=False)
    b = WRTDSK(fitted_base_model, niter=3, seed=np.random.default_rng(7)).filter(df_wq, df_q, verbose=False)

    assert np.random.random() == expected_global  # global state untouched
    pd.testing.assert_series_equal(a["ConcKalman"], b["ConcKalman"])
