import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import truncnorm

from wrtds_py.models.classic import WRTDS


def _ar1_gap_residuals(z_left: float, z_right: float, gap_days: int, rho: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Draw the standardized residuals on the days strictly between two samples.

    `gap_days` is the number of days from the left sample to the right one. The draw is an
    expected residual, set by the distance to each sample, plus a random AR(1) deviation.
    """
    k = np.arange(1, gap_days) # days since the left sample

    # Expected residual: weights of the left and right samples. Each weight decays with the
    # distance to its sample, and both vanish in the middle of long gaps (plain WRTDS)
    denom = 1.0 - rho ** (2 * gap_days)
    w_left = (rho ** k - rho ** (2 * gap_days - k)) / denom
    w_right = (rho ** (gap_days - k) - rho ** (gap_days + k)) / denom
    expected = w_left * z_left + w_right * z_right

    # Random deviation: simulate a free AR(1) path across the gap and remove the part that
    # its own end points explain, leaving noise that is zero at both samples
    path = np.empty(gap_days + 1)
    path[0] = rng.standard_normal()
    innovations = rng.standard_normal(gap_days) * np.sqrt(1.0 - rho ** 2)
    for t in range(1, gap_days + 1):
        path[t] = rho * path[t - 1] + innovations[t - 1]
    deviation = path[1:gap_days] - (w_left * path[0] + w_right * path[gap_days])

    return expected + deviation

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
        n_sampled = len(sample_days)

        yhat_q = df_q_pred['yHat'].values
        se_q = df_q_pred['SE'].values
        n_record = day_q.max() + 1 # days from the first to the last day of the record

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

                # Standardized residual for every day of the record. Before the first and after
                # the last sample there is no information, so they stay N(0, 1): plain WRTDS
                z = rng.standard_normal(n_record)
                z[sample_days] = z_days

                # Each gap between two consecutive samples is informed by both of them
                for i in range(n_sampled - 1):
                    left, right = sample_days[i], sample_days[i + 1]
                    z[left + 1 : right] = _ar1_gap_residuals(z_days[i], z_days[i + 1], right - left, self.rho, rng)

                # Rescale the standardized residuals by each day's SE and transform back to concentration.
                # The Monte Carlo average provides the bias correction, so no exp(sigma^2 / 2) is applied.
                conc += np.exp(yhat_q + z[day_q] * se_q)

                pbar.update(1)

        # Average the niter samples to get the final estimates
        mean_conc = conc / self.niter
        
        df_out = df_q_pred.copy()
        df_out['ConcKalman'] = mean_conc
        
        self._is_fitted = True
        return df_out
