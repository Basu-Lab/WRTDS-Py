import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import lfilter
from scipy.stats import truncnorm

from wrtds_py.models.classic import WRTDS

class WRTDSK:
    def __init__(self, base_model: WRTDS, rho: float = 0.9, niter: int = 200,
                 seed: int | np.random.Generator | None = None):
        if not getattr(base_model, '_is_fitted', False):
            raise ValueError("The base WRTDS model must be fitted before running the Kalman filter.")
        if not 0 <= rho < 1:
            raise ValueError("`rho` must be in [0, 1).")
        
        self.base_model = base_model
        self.rho = rho
        self.niter = niter
        self.seed = seed
        self._is_fitted = False

    def filter(self, df_wq: pd.DataFrame, df_q: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        # Local generator: the global NumPy random state is left untouched
        rng = np.random.default_rng(self.seed)

        df_wq_clean, df_q_clean = self.base_model._prepare_data(df_wq, df_q)

        # Get predictions from the base WRTDS model
        df_wq_pred = self.base_model.predict(df_wq_clean)
        df_q_pred = self.base_model.predict(df_q_clean)

        # Prepare arrays for the AR(1) model
        y_obs = df_wq_pred['LogConc'].values # log-observations
        y_hat = df_wq_pred['yHat'].values # log-predictions
        se = df_wq_pred['SE'].values # standard errors
        uncen = df_wq_pred['Uncen'].values # 1 if uncensored, 0 if left-censored
        
        n_wq = len(df_wq_pred)
        n_q = len(df_q_pred)

        conc = np.zeros(n_q) # array to store samples

        # Ensure standard errors are strictly positive and finite
        se_safe = np.where(np.isfinite(se) & (se > 1e-4), se, 1e-4)

        # Standardized residuals for the AR(1) model, as in EGRET's WRTDSKalman:
        # z = (ln c - yHat) / SE, rescaled later by each day's own SE
        residuals = (y_obs - y_hat) / se_safe

        censored_mask = (uncen == 0)
        num_censored = np.sum(censored_mask)

        if num_censored > 0:
            # Truncation bounds for the truncated normal distribution for censored data
            a_vec = np.full(num_censored, -np.inf) # Smallest possible value for left-censor data
            b_vec = residuals[censored_mask] # Standardized difference between 'detection limit' and predicted value

            # Guard against NaN/inf values in the residuals themselves
            b_vec = np.nan_to_num(b_vec, nan=0.0, posinf=10.0, neginf=-10.0)

        # Whole-day counts since the start of the daily record
        origin = df_q_pred.index.min().normalize()
        day_q = (df_q_pred.index.normalize() - origin).days.to_numpy()
        day_wq = (df_wq_pred.index.normalize() - origin).days.to_numpy()

        # Days with at least one sample; `sample_of` maps each sample to its day in `sample_days`
        sample_days, sample_of = np.unique(day_wq, return_inverse=True)
        n_days = len(sample_days)

        # Classify every day of the record relative to the sampled days
        left = np.searchsorted(sample_days, day_q, side='right') - 1 # last sampled day on or before
        is_sample = (left >= 0) & (sample_days[np.clip(left, 0, None)] == day_q)
        is_gap = ~is_sample & (left >= 0) & (left < n_days - 1) # strictly between two sampled days
        is_outside = ~is_sample & ~is_gap # before the first or after the last sampled day

        # AR(1) bridge weights: conditional mean of a unit-variance AR(1) at k days into a
        # gap of N days, given its values at both ends, is w_l * z_0 + w_r * z_N
        gap_l = left[is_gap]
        gap_r = gap_l + 1
        k = (day_q[is_gap] - sample_days[gap_l]).astype(float)
        n = (sample_days[gap_r] - sample_days[gap_l]).astype(float)
        rho = self.rho
        denom = 1.0 - rho ** (2 * n)
        w_l = (rho ** k - rho ** (2 * n - k)) / denom
        w_r = (rho ** (n - k) - rho ** (n + k)) / denom

        # Daily AR(1) path covering every day of the record (stationary, unit variance)
        n_record = day_q.max() + 1
        innovation_sd = np.sqrt(1.0 - rho ** 2)

        yhat_q = df_q_pred['yHat'].values
        se_q = df_q_pred['SE'].values

        # Sampling residuals
        with tqdm(total=self.niter, disable=not verbose, desc="Running WRTDS-K", ascii=True) as pbar:
            for _ in range(self.niter):
                residuals_iter = residuals.copy() # base residuals
                if num_censored > 0: # censored data
                    samples = truncnorm.rvs(a_vec, b_vec, size=num_censored, random_state=rng) # samples for censored data
                    residuals_iter[censored_mask] = samples # already standardized

                # One sample per sampled day, chosen at random when a day has several
                order = rng.permutation(n_wq)
                _, first = np.unique(sample_of[order], return_index=True)
                z_days = residuals_iter[order[first]]

                # Unconditional AR(1) path: v[0] ~ N(0, 1), v[t] = rho * v[t-1] + sqrt(1 - rho^2) * eps[t]
                eps = rng.standard_normal(n_record)
                eps[0] /= innovation_sd
                path = lfilter([innovation_sd], [1.0, -rho], eps)
                path_days = path[sample_days]

                z = np.empty(n_q)
                z[is_sample] = z_days[left[is_sample]]
                # Condition the path on both ends of each gap: an exact draw from the AR(1) bridge
                z[is_gap] = (path[day_q[is_gap]]
                             + w_l * (z_days[gap_l] - path_days[gap_l])
                             + w_r * (z_days[gap_r] - path_days[gap_r]))
                # Before the first and after the last sample there is no information: plain WRTDS
                z[is_outside] = rng.standard_normal(is_outside.sum())

                # Rescale the standardized residuals by each day's SE and transform back to concentration.
                # The Monte Carlo average provides the bias correction, so no exp(sigma^2 / 2) is applied.
                conc += np.exp(yhat_q + z * se_q)

                pbar.update(1)

        # Average the niter samples to get the final estimates
        mean_conc = conc / self.niter
        
        df_out = df_q_pred.copy()
        df_out['ConcKalman'] = mean_conc
        
        self._is_fitted = True
        return df_out
