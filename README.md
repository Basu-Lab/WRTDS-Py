# wrtds-py

[![PyPI version](https://img.shields.io/pypi/v/wrtds-py.svg)](https://pypi.org/project/wrtds-py/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`wrtds-py` is a Python implementation of the **Weighted Regressions on Time, Discharge, and Season (WRTDS)** model, including the **WRTDS-Kalman (WRTDS-K)** autoregressive filter. 

It is adapted from the official USGS [`EGRET`](https://github.com/DOI-USGS/EGRET) R package.

---

## Status

**Current Version: 1.0.0**

`wrtds-py` is stable and validated against the USGS `EGRET` R package (v3.0.11) on its reference example (the Choptank River). Releases are published on [PyPI](https://pypi.org/project/wrtds-py/) and archived on Zenodo. See [CHANGELOG.md](https://github.com/Basu-Lab/WRTDS-Py/blob/main/CHANGELOG.md) for the release history.

## Key Features

* **Classic WRTDS (`WRTDS`):** Pre-computes the prediction surface. Log-concentration, standard error and concentration are calculated using linear regression where the coefficients are estimated using maximum likelihood considering left-censored data.
* **WRTDS-Kalman (`WRTDSK`):** Implements a filtering method that smoothes the residuals using an AR(1) process.

> Note: WRTDS-Kalman is not actually a Kalman filter in the strictest sense. In Zhang and Hirsch (2019) the authors derive this approach from the theory of Kalman fitering, but a more accurate description is AR(1) smoothing with Monte Carlo simulations for censored samples.


## Background

For a detailed explanation of the underlying statistical methods, including the design matrix of the linear model, left-censored (Tobit) maximum likelihood estimation, weighting functions, and the AR(1) smoother, see [docs/methodology.md](https://github.com/Basu-Lab/WRTDS-Py/blob/main/docs/methodology.md).


## Installation

`wrtds-py` requires Python 3.12 or newer and is available on [PyPI](https://pypi.org/project/wrtds-py/).

### Using `uv` (Recommended)

```bash
uv add wrtds-py
```

### Using standard `pip`

```bash
pip install wrtds-py
```

### From source

```bash
git clone https://github.com/Basu-Lab/WRTDS-Py
cd WRTDS-Py
uv sync
```

---

## Quickstart
Also have a look at the [example Jupyter notebook](https://github.com/Basu-Lab/WRTDS-Py/blob/main/docs/example.ipynb) for a more detailed demonstration.

### 1. Classic WRTDS

```python
import pandas as pd
from wrtds_py import WRTDS

# Load discharge (df_q) and water quality (df_wq) data
# DataFrames must have a DatetimeIndex
df_q = pd.read_csv("discharge.csv", parse_dates=["Date"], index_col="Date")
df_wq = pd.read_csv("water_quality.csv", parse_dates=["Date"], index_col="Date")

# Initialize and fit the base WRTDS model
model = WRTDS(
    window_time=7.0,       # Half-window for time (years)
    window_discharge=2.0,  # Half-window for discharge (log-units)
    window_season=0.5,     # Half-window for season (years)
    min_num_obs=50,
    min_num_uncen=30
)

model.fit(df_wq=df_wq, df_q=df_q)

# Generate daily predictions
df_pred = model.predict(df_q)
print(df_pred[["yHat", "SE", "ConcHat"]].head())
```

### 2. WRTDS-Kalman (Autoregressive Filter)

```python
from wrtds_py import WRTDSK

# Pass the fitted base model into the Kalman filter
kalman = WRTDSK(
    base_model=model,
    rho=0.9,       # Daily correlation coefficient (decay factor)
    niter=100,     # Monte Carlo iterations for censored samples
    seed=42        # Seed for reproducibility
)

# Run the filter on the daily discharge record
df_kalman = kalman.filter(df_wq=df_wq, df_q=df_q)
print(df_kalman[["ConcHat", "ConcKalman"]].head())
```

## Testing & Benchmarking

To run the test suite from a source checkout:

```bash
uv run pytest
```

`tests/test_r_reference.py` validates the models against USGS `EGRET` v3.0.11 by running it live through `rpy2`, so it requires a working R installation with the `EGRET` package installed. To run only the pure-Python tests:

```bash
uv run pytest tests/test_components.py tests/test_models.py
```

## Citation

If you use `wrtds-py` in your research, please cite it using the metadata in [CITATION.cff](https://github.com/Basu-Lab/WRTDS-Py/blob/main/CITATION.cff) (GitHub's "Cite this repository" button generates APA and BibTeX entries from it):

> Álvarez Chaves, M. (2026). *wrtds-py: Weighted Regressions on Time, Discharge, and Season in Python* (Version 1.0.0) [Computer software]. https://doi.org/10.5281/zenodo.XXXXXXX <!-- TODO: replace with the Zenodo DOI -->

Please also cite the original WRTDS and WRTDS-K publications listed under [References](#references).

## License

Distributed under the MIT License. See [`LICENSE`](https://github.com/Basu-Lab/WRTDS-Py/blob/main/LICENSE) for more information.

## References

* **Hirsch, R. M., Moyer, D. L., & Archfield, S. A. (2010).** *Weighted Regressions on Time, Discharge, and Season (WRTDS), with an Application to Chesapeake Bay River Inputs*. JAWRA. [https://doi.org/10.1111/j.1752-1688.2010.00482.x](https://doi.org/10.1111/j.1752-1688.2010.00482.x)
* **Zhang, Q., & Hirsch, R. M. (2019).** *River Water-Quality Concentration and Flux Estimation Can be Improved by Accounting for Serial Correlation Through an Autoregressive Model*. Water Resources Research. [https://doi.org/10.1029/2019WR025338](https://doi.org/10.1029/2019WR025338)
* **Hirsch, R., DeCicco, L., & Murphy, J. (2025).** *Exploration and Graphics for RivEr Trends (EGRET)*. U.S. Geological Survey. [https://pubs.usgs.gov/tm/04/a10/](https://pubs.usgs.gov/tm/04/a10/)