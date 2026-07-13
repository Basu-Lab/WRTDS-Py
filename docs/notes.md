# WRTDS

## 1. What is WRTDS?

**Weighted Regressions on Time, Discharge, and Season (WRTDS)** is statistical model designed to estimate water quality concentrations and mass flux (load) in rivers.

WRTDS is a localized, non-parametric model. Using continous daily discharge data and sparse water-quality samples, it builds a regression surface that can be used to predict concetrations for any 'time', 'discharge' and 'season' combination.

The `wrtds_py` package is a Python implementation of the WRTDS algorithm as implmented in the [`EGRET`](https://github.com/DOI-USGS/EGRET) R package developed by the U.S. Geological Survey.

**References**:

* Hirsch, R., DeCicco, L., & Murphy, J. (2025). *Exploration and Graphics for RivEr Trends (EGRET)*. U.S. Geological Survey. https://pubs.usgs.gov/tm/04/a10/
* Hirsch, R. M., Moyer, D. L., & Archfield, S. A. (2010). *Weighted Regressions on Time, Discharge, and Season (WRTDS), with an Application to Chesapeake Bay River Inputs*. Journal of the American Water Resources Association, 46(5), 857–880. https://doi.org/10.1111/j.1752-1688.2010.00482.x
* Zhang, Q., & Hirsch, R. M. (2019). *River Water-Quality Concentration and Flux Estimation Can be Improved by Accounting for Serial Correlation Through an Autoregressive Model*. Water Resources Research, 55(11), 9705–9723. https://doi.org/10.1029/2019WR025338


---

## 2. Data

For this initial implementation we use very similar data structures to those used in the original EGRET R package and the whole USGS data ecosystem. Instead of an `eList` object, we use two separate DataFrames: `df_daily` and `df_sample`.

### `df_daily` (Continuous Discharge Record)

| Column | Description |
| --- | --- |
| **`Date`** | pandas `DatetimeIndex` representing every continuous day. |
| **`Q`** | Raw daily mean streamflow (discharge) in any units (cfs, m³/s). |
| **`LogQ`** | Natural logarithm of streamflow ($\ln(Q)$). |
| **`DecYear`** | Decimal year representation of the date (e.g., 2010.5 for July 1). This is the `time` variable. |

### `df_sample` (Sparse Water Quality Record)

| Column | Description |
| --- | --- |
| **`ConcLow`** | Lower boundary of the recorded concentration. For uncensored data, this equals the recorded value. For censored data (e.g. reported as `< 0.05`), this is `0.0` or `NaN`. |
| **`ConcHigh`** | Upper boundary of the recorded concentration. For uncensored data, equals `ConcLow`. For censored data, equals the detection limit (e.g., `0.05`). |
| **`Uncen`** | Binary flag (`1` = true detection, `0` = censored/below detection limit). |
| **`DecYear`** | Decimal year of the sample date. |

---

## 3. The Estimation Grid (`surfaces_`)

The whole point of WRTDS is to pre-compute a 3D grid of discharge, time and concentration. Predictions are generated later via interpolation.

* **Shape:** `(14, N, 3)`
* **Dimension 1 (Discharge):** 14 equally spaced $\ln(Q)$ nodes spanning the minimum to maximum historical daily flows.
* **Dimension 2 (Time/Season):** $N$ time nodes, strictly stepped at exactly 16 increments per calendar year to perfectly capture seasonal cycles without overfitting.
* **Dimension 3 (Outputs):** `[yHat, SE, ConcHat]` for every intersection in the grid.

---

## 4. The Weighting System

For a specific target node in the grid, every historical sample is assigned a weight based on its distance in time, discharge, and season using the **Tri-Cube Weight Function**:

$$w = \begin{cases} \left(1 - \left(\frac{d}{h}\right)^3\right)^3 & \text{if } d \le h \\ 0 & \text{if } d > h \end{cases}$$

Where $d$ is the distance and $h$ is the half-window bandwidth. The total weight applied to a sample is the product of all three dimensions: $W = w_{time} \cdot w_{discharge} \cdot w_{season}$. Seasonality is calculated cyclically, with a maximum possible distance of 0.5 years.

### Adaptive Window Expansion

If a grid node lacks sufficient data to run a robust regression (e.g., during extreme historical floods), the algorithm dynamically adapts:

1. **Edge Adjustment:** Pre-expands the time window asymmetrically if the node is at the very beginning or end of the historical record.
2. **Iterative Expansion:** Multiplies the spatial windows by 1.1 iteratively until the minimum observation thresholds (`min_num_obs`, `min_num_uncen`) are met.

---

## 5. The Regression Engine (Tobit MLE)

Because water quality data includes censored "less-than" values, Ordinary Least Squares (OLS) produces mathematically biased results. `wrtds_py` implements **Adjusted Maximum Likelihood Estimation (AMLE)** via an interval-censored Tobit survival regression.

### Centered Design Matrix

To dramatically optimize the math, the local matrix is centered on the target node. This forces $(t_{sample} - t_{target}) = 0$ at the point of prediction, meaning the prediction (`yHat`) simplifies entirely to the intercept and the seasonal terms:

$$y = \beta_0 + \beta_1(t - t_{tgt}) + \beta_2(\ln(Q) - \ln(Q)_{tgt}) + \beta_3\sin(2\pi t) + \beta_4\cos(2\pi t) + \epsilon$$

### Maximum Likelihood Optimization

The custom negative log-likelihood function (optimized via `L-BFGS-B`) treats uncensored data as points on a normal Probability Density Function (PDF), and censored data as cumulative probabilities (CDF) bounded by the detection limit:

* **Uncensored Log-Likelihood:** $W \cdot \ln(\text{PDF}(z) / \sigma)$
* **Censored Log-Likelihood:** $W \cdot \ln(\text{CDF}(z))$

---

## 6. Back-Transformation (Bias Correction)

Because WRTDS models the natural logarithm of concentration, exponentiating the result directly back to real units ($mg/L$) introduces a severe low bias.

While early versions of the algorithm used Naihua Duan's non-parametric smearing estimator, interval-censored data prevents the exact calculation of residuals. Therefore, `wrtds_py` uses the mathematically rigorous, modern EGRET standard of evaluating the **parametric expected value** of the log-normal distribution:

$$\hat{C} = \exp\left(\hat{y} + \frac{\sigma^2}{2}\right)$$

Here is a clean, structured section for WRTDS-K that you can append directly to the bottom of your `notes.md` file. I have kept the style concise, modular, and focused on the core mechanics, matching the structure we used when breaking down the classic WRTDS math.

---

## WRTDS-Kalman (WRTDS-K)

While classic WRTDS is excellent at capturing long-term trends and smooth seasonal averages, it lacks short-term memory. If a river experiences an acute pollution event on Tuesday, WRTDS-K helps model the likelihood that it is still somewhat polluted on Wednesday. It achieves this by applying an autoregressive filter to the residuals of the base WRTDS model.

### 1. The Core Concept

WRTDS-K improves daily predictions by calculating the difference (residual) between the actual water quality observations and the base WRTDS predictions, and then interpolating those residuals across the unobserved days between samples.

### 2. Handling Censorship (Monte Carlo Simulation)

Because water quality data often contains left-censored values (measurements below the laboratory detection limit), we don't always have an exact observation to calculate a perfect residual. WRTDS-K solves this probabilistically:

* **Uncensored Data:** The residual is exact: $e = y_{obs} - \hat{y}$.
* **Censored Data:** WRTDS-K runs a Monte Carlo simulation (`niter` times, typically 100). In each iteration, it draws a random "true" value from a truncated normal distribution to compute a possible residual.

### 3. The AR(1) Process (Short-Term Memory)

For every single day in the continuous streamflow record, the filter looks at the nearest preceding and succeeding water quality sample residuals.

* It uses a continuous-time Autoregressive (AR(1)) process to interpolate the expected residual for that day.
* The correlation decays exponentially over time, controlled by the parameter $\rho$ (typically set to **0.9**).
* This means the "memory" or influence of a specific sampled residual decays by **10%** for each day that passes.

### 4. Final Assembly

For each of the Monte Carlo iterations, the algorithm performs the following assembly:

1. **Recombination:** The interpolated daily residuals are added back to the daily base WRTDS predictions in log-space.
2. **Variance Correction:** Because the AR(1) process explains some of the variance, the standard error is adjusted:

$$\sigma_{adj} = \text{SE} \cdot \sqrt{\max(0, 1 - \rho^2)}$$


3. **Transformation:** The results are exponentiated back into real space to calculate concentration:

$$\text{Conc} = \exp\left(\hat{y}_{adj} + \frac{\sigma_{adj}^2}{2}\right)$$



Finally, the estimated daily concentrations are averaged across all 100 iterations to produce the final **Generated Concentration (`GenConc`)** and **Generated Flux (`GenFlux`)**.