# WRTDS

## 1. What is WRTDS?

**Weighted Regressions on Time, Discharge, and Season (WRTDS)** is statistical model designed to estimate water quality concentrations and mass flux (load) in rivers.

WRTDS is a localized, non-parametric model. Using continuous daily discharge data and sparse water-quality samples, it builds a regression surface that can be used to predict concentrations for any 'time', 'discharge' and 'season' combination.

The `wrtds_py` package is a Python implementation of the WRTDS algorithm similar to that implemented in the [`EGRET`](https://github.com/DOI-USGS/EGRET) R package developed by the U.S. Geological Survey.

**References**:

* Hirsch, R. M., Moyer, D. L., & Archfield, S. A. (2010). *Weighted Regressions on Time, Discharge, and Season (WRTDS), with an Application to Chesapeake Bay River Inputs*. Journal of the American Water Resources Association, 46(5), 857–880. https://doi.org/10.1111/j.1752-1688.2010.00482.x
* Hirsch, R., DeCicco, L., & Murphy, J. (2025). *Exploration and Graphics for RivEr Trends (EGRET)*. U.S. Geological Survey. https://pubs.usgs.gov/tm/04/a10/
* Zhang, Q., & Hirsch, R. M. (2019). *River Water-Quality Concentration and Flux Estimation Can be Improved by Accounting for Serial Correlation Through an Autoregressive Model*. Water Resources Research, 55(11), 9705–9723. https://doi.org/10.1029/2019WR025338


---

## 2. Data

For this Python version we use very similar data structures to those used in the original EGRET R package and the whole USGS data ecosystem. Instead of an `eList` object, we use two separate DataFrames: `df_q` and `df_wq`.

### `df_q`: Continuous Discharge Record

| Column | Description |
| --- | --- |
| **`Date`** | pandas `DatetimeIndex` representing every continuous day. |
| **`Q`** | Raw daily mean streamflow (discharge) in any units (e.g. cfs, m³/s). |
| **`LogQ`** | Natural logarithm of streamflow ($\ln(Q)$). Computed automatically from `Q` if not supplied. |
| **`DecYear`** | Decimal year representation of the date (e.g., 2010.5 for July 1). This is the `time` variable. Computed automatically from the index if not supplied. |

### `df_wq`: Sparse Water Quality Record

| Column | Description |
| --- | --- |
| **`Conc`** | The recorded concentration (water quality feature). For censored data (e.g. reported as `< 0.05`), this is the detection limit itself (e.g. `0.05`), not the true (unknown) value. |
| **`Uncen`** | Binary flag (`1` = true detection, `0` = censored/below detection limit). |
| **`LogQ`** | Natural logarithm of streamflow at the sample date. If not supplied directly, it is mapped from `df_q` by matching on date. Every sample date must therefore exist in `df_q`'s index. |
| **`DecYear`** | Decimal year of the sample date. Computed automatically from the index if not supplied. |

Note: unlike `EGRET`, which represents censored samples with a `ConcLow`/`ConcHigh` interval, `wrtds_py` only supports a single left-censoring bound (i.e. the detection limit) via `Conc` + `Uncen`.

---

## 3. The Estimation Grid (`surface_`)

The whole point of WRTDS is to pre-compute a 3D grid of discharge, time and concentration. Predictions are generated later via interpolation.

* **`surface_` shape:** (discharge, time, outputs)
* **Dimension 1 (Discharge):** Typically 14 equally spaced $\ln(Q)$ nodes spanning the minimum to maximum historical daily flows, stored in `grid_logq_`.
* **Dimension 2 (Time/Season):** $N$ time nodes, stepped at exactly 1/16 of a calendar year to capture seasonal cycles, stored in `grid_time_`. $N$ is `max_time` minus `min_time` divided by 1/16, rounded up to the nearest integer.
* **Dimension 3 (Outputs):** `[yHat, SE, ConcHat]` for every node in the grid. `yHat` is the predicted log-concentration, `SE` is the standard error of the prediction, and `ConcHat` is the bias-corrected concentration in real units (i.e. $mg/L$).

The effective (possibly expanded — see below) window used to fit each node is stored separately in `effective_windows_`, with the same `(14, N, 3)` shape but holding `[window_time, window_discharge, window_season]` instead.


## 4. The Weighting System

For a specific target node in the grid, every historical sample is assigned a weight based on its distance in time, discharge, and season using the **Tri-Cube Weight Function**:

$$w = \begin{cases} \left(1 - \left(\frac{d}{h}\right)^3\right)^3 & \text{if } d \le h \\ 0 & \text{if } d > h \end{cases}$$

Where $d$ is the distance and $h$ is the half-window bandwidth. The total weight applied to a sample is the product of all three dimensions: $W = w_{time} \cdot w_{discharge} \cdot w_{season}$.

The default half-window bandwidths are:
* **`window_time`** = 7 years
* **`window_discharge`** = 2 log-units
* **`window_season`** = 0.5 years (i.e. 6 months)

If a sample is outside the half-window in any dimension, it receives a weight of zero and is excluded from the regression. 

If a node is very close to the edge of the historical record (e.g., the first or last year of data), the algorithm automatically pre-expands the time window asymmetrically to ensure that enough samples are included in the regression.

If a node has too few samples to run a regression, the algorithm automatically expands the half-window bandwidths until a minimum number of observations is met i.e. `min_num_obs` and `min_num_uncen`. This is done in incremental steps of $10\%$.

## 5. Weighted Linear Regression with Censored Data

Because water quality data includes censored "less-than" values (i.e. detection limits), Ordinary Least Squares (OLS) would produce mathematically biased results. `wrtds_py` implements a **left-censored regression** also known as **Tobit regression**, solved via Maximum Likelihood Estimation (MLE).

`EGRET` supports a more general interval-censored regression (`ConcLow`/`ConcHigh`) but this is not implemented in `wrtds_py`. We only support left-censored regression because it is the most common case in water quality data.

### Design Matrix

To optimize the math, the design matrix is centered on the target node. This forces $(t_{active} - t_{target}) = 0$ at the point of prediction, meaning the prediction (`yHat`) simplifies entirely to the intercept and the seasonal terms. Yet the general regression equation is still:

$$y = \beta_0 + \beta_1(t_{active} - t_{target}) + \beta_2(\ln(Q)_{active} - \ln(Q)_{target}) + \beta_3\sin(2\pi t_{active}) + \beta_4\cos(2\pi t_{active}) + \epsilon$$

In matrix form, this is:
$$\mathbf{y} = \mathbf{X}\beta + \epsilon$$

Where $\mathbf{y}$ is the vector of log-concentrations, $\mathbf{X}$ is the design matrix, $\beta$ is the vector of coefficients, and $\epsilon$ is the error term.

An initial guess for the coefficients is generated via a weighted least squares (WLS) regression on all active samples, with censored samples taken at their detection limit. This is used as a starting point for the MLE optimization.

$$\hat{\beta} = (\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W} \mathbf{y}$$

Where $\mathbf{W}$ is the diagonal matrix of weights for each sample, rescaled so that the weights average to 1. The initial $\sigma$ is the weighted root mean square of the WLS residuals, floored at 0.05.

### Maximum Likelihood

The weighted negative log-likelihood is minimized over $(\beta, \sigma)$ with `L-BFGS-B`, with $\sigma$ bounded below by $10^{-5}$. With the standardized residual

$$z_i = \frac{y_i - \mathbf{x}_i^T\beta}{\sigma}$$

uncensored samples contribute the log of the Gaussian Probability Density Function (PDF), and censored samples the log of the Cumulative Distribution Function (CDF) up to the detection limit, i.e. the probability that the true value lies below it:

| Sample | Log-likelihood $\ell_i$ |
| --- | --- |
| Uncensored | $\ln\phi(z_i) - \ln\sigma = -\frac{z_i^2}{2} - \ln\sqrt{2\pi} - \ln\sigma$ |
| Censored | $\ln\Phi(z_i)$ |

$$-\ln L(\beta, \sigma) = -\sum_i w_i \, \ell_i$$

$\ln\Phi$ is evaluated with `scipy.special.log_ndtr`, which stays accurate far into the lower tail, where $\Phi$ itself underflows to zero and a naive `log(norm.cdf(z))` would return $-\infty$.

#### Analytic Gradient

The objective returns its analytic gradient together with its value (`jac=True`), so the optimizer never relies on finite differences. Using $\partial z_i / \partial\beta = -\mathbf{x}_i / \sigma$ and $\partial z_i / \partial\sigma = -z_i / \sigma$:

| Sample | $\partial\ell_i / \partial z_i$ | $\partial\ell_i / \partial\sigma$ |
| --- | --- | --- |
| Uncensored | $-z_i$ | $\dfrac{z_i^2 - 1}{\sigma}$ |
| Censored | $\lambda(z_i)$ | $-\dfrac{\lambda(z_i)\, z_i}{\sigma}$ |

Where $\lambda(z) = \phi(z) / \Phi(z)$ is the inverse Mills ratio. To keep it stable in the lower tail, it is computed in log space as $\lambda(z) = \exp\left(\ln\phi(z) - \ln\Phi(z)\right)$ rather than as a ratio of two numbers that both underflow. The gradient of the negative log-likelihood is then:

$$\frac{\partial(-\ln L)}{\partial\beta} = \sum_i \frac{w_i}{\sigma} \frac{\partial\ell_i}{\partial z_i} \mathbf{x}_i \qquad \frac{\partial(-\ln L)}{\partial\sigma} = -\sum_i w_i \frac{\partial\ell_i}{\partial\sigma}$$

The analytic gradient is not only faster but required for convergence. When a node has no censored samples, the WLS starting point is already the exact optimum. Finite-difference gradients at that point are dominated by rounding noise, the line search cannot make progress, and `L-BFGS-B` ends with `ABNORMAL` status. Since a failed optimization returns NaN for the node, this previously left ~9% of the nodes in `surface_` as NaN. With the exact gradient, the optimizer recognizes the starting point as converged. `test_tobit_gradient` checks the analytic gradient against finite differences.

## 6. Back-Transformation (Bias Correction)

Because WRTDS models log-concentration, exponentiating the result directly back to real units ($mg/L$) introduces a low bias. To correct for this, we back-transform the log-predictions using the expected value of a log-normal distribution:

$$\hat{C} = \exp\left(\hat{y} + \frac{\sigma^2}{2}\right)$$


# WRTDS-Kalman (WRTDS-K)

While classic WRTDS is excellent at capturing long-term trends and smooth seasonal averages, it lacks short-term memory. If a river experiences an acute pollution event on Tuesday, introducing a Kalman filter helps model the likelihood that the river is still somewhat polluted on Wednesday. This is achieved by applying an autoregressive filter `AR(1)` to the residuals of the base WRTDS model.

### 1. The Core Idea

WRTDS-K improves daily predictions by calculating the difference (residual) between the actual water quality observations and the base WRTDS predictions, and then interpolating those residuals across the unobserved days between samples.

### 2. Handling Censored Data

Because water quality data often contains left-censored values (measurements below the laboratory detection limit), we don't always have an exact observation to calculate a perfect residual. WRTDS-K solves this probabilistically:

* **Uncensored Data:** The residual is exact: $e = y_{obs} - \hat{y}$.
* **Censored Data:** WRTDS-K runs a Monte Carlo simulation (`niter` times, typically 100). In each iteration, it draws a random "true" value from a truncated normal distribution to compute a possible residual.

### 3. The AR(1) Process

Having a set of residuals for water quality data, for every single day in the continuous streamflow record, the filter looks at the nearest preceding and succeeding residuals.

The filter uses an autoregressive process `AR(1)` to interpolate the expected residual for each day. The correlation decays exponentially over time, controlled by the parameter $\rho$ (typically set to **0.9**). This means the "memory" or influence of a specific sampled residual decays by **10%** for each day that passes.

As an example, for a previously calculated concentration value at time $t_{50}$ we find that the closest residuals are at $t_{40}$ and $t_{70}$. Therefore we can count the number of days between the target day and the nearest residuals: `days_l` = 10 and `days_r` = 20. Now, for these residuals we calculate their weights using the `AR(1)` decay function: `w_l` = $\rho^{\text{days\_l}}$ and `w_r` = $\rho^{\text{days\_r}}$. Finally, we calculate the expected residual for $t_{50}$ as a weighted average of the two nearest residuals:

$$e_{t_{50}} = \frac{w_l e_{t_{40}} + w_r e_{t_{70}}}{w_l + w_r}$$

And add it back to the base WRTDS prediction to get the final Kalman-adjusted prediction.

$$\hat{y}_{adj} = \hat{y} + e_{t_{50}}$$

Where $\hat{y}_{adj}$ is the adjusted prediction in log-space. The standard error is also adjusted to account for the uncertainty introduced by the AR(1) process. The standard error of the adjusted prediction is calculated as:

$$\sigma_{adj} = \text{SE} \cdot \sqrt{\max(0, 1 - \rho^2)}$$

Finally, the adjusted log-prediction is back-transformed to real units using the same bias correction as classic WRTDS:

$$\hat{C} = \exp\left(\hat{y}_{adj} + \frac{\sigma_{adj}^2}{2}\right)$$

For the censored data, the estimated daily concentrations are averaged across all `niter` iterations to produce the final output. These new predictions are stored in the **`ConcKalman`** column.