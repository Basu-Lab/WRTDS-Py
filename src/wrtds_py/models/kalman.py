import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import truncnorm

from wrtds_py.base import BaseWRTDS
from wrtds_py.models.classic import WRTDS

class WRTDSK:
    def __init__(self, base_model: WRTDS, rho: float = 0.9, niter: int = 100, seed: int | None = None):
        if not getattr(base_model, '_is_fitted', False):
            raise ValueError("The base WRTDS model must be fitted before running the Kalman filter.")
        
        self.base_model = base_model
        self.rho = rho
        self.niter = niter
        self.seed = seed
        self._is_fitted = False

    def filter(self, df_wq: pd.DataFrame, df_q: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        if self.seed is not None:
            np.random.seed(self.seed)

        df_wq_clean, df_q_clean = self.base_model._validate_data(df_wq, df_q)

        # Get predictions from the base WRTDS model
        df_wq_pred = self.base_model.predict(df_wq_clean)
        df_q_pred = self.base_model.predict(df_q_clean)

        # Prepare arrays for the AR(1) model
        y_obs = df_wq_pred['LogConc'].values # log-observations
        y_hat = df_wq_pred['yHat'].values # log-predictions
        se = df_wq_pred['SE'].values # standard errors
        uncen = df_wq_pred['Uncen'].values # 1 if uncensored, 0 if left-censored
        
        time_wq = df_wq_pred['DecYear'].values
        time_q = df_q_pred['DecYear'].values
        
        n_wq = len(df_wq_pred)
        n_q = len(df_q_pred)
        
        conc = np.zeros(n_wq) # array to store samples
        
        # 3. Monte Carlo Iterations
        with tqdm(total=self.niter, disable=not verbose, desc="Running WRTDS-K", ascii=True) as pbar:
            for _ in range(self.niter):
                # a) Generate residuals for this iteration
                residuals_iter = np.zeros(n_wq)
                
                for i in range(n_wq):
                    if uncen[i] == 1:
                        # Exact residual for uncensored data
                        residuals_iter[i] = y_obs[i] - y_hat[i]
                    else:
                        # Draw a possible residual from a truncated normal distribution
                        # Log-space
                        a = -np.inf # Smallest possible value for left-censor data
                        b = (y_obs[i] - y_hat[i]) # Difference between 'detection limit' and predicted value
                        b = b / se[i] # truncnorm expects z-scores, so we use the standard error to scale
                        draw = truncnorm.rvs(a, b) # 1 sample
                        residuals_iter[i] = draw * se[i] # Re-scale back to log-space
                
                # Generate residuals for 'q' data using the AR(1) model
                daily_res_interp = np.zeros(n_q)
                
                # Find the closest indices in 'wq' for each 'q' time point
                left_idx = np.searchsorted(time_wq, time_q, side='right') - 1
                right_idx = np.clip(left_idx + 1, 0, n_wq - 1)
                
                for k in range(n_q):
                    idx_l, idx_r = left_idx[k], right_idx[k]
                    if idx_l < 0:
                        idx_l = 0 # Handle edge case before first sample
                    
                    # Count the days between the 'q' time point and the closest 'wq' observation
                    days_l = (time_q[k] - time_wq[idx_l]) * 365.25
                    days_r = (time_wq[idx_r] - time_q[k]) * 365.25
                    
                    if days_l < 0:
                        days_l = np.inf
                    if days_r < 0:
                        days_r = np.inf
                    
                    # AR(1) weighting based on the time difference
                    # More days away means more decay in the influence of the residual
                    w_l = self.rho ** days_l if days_l != np.inf else 0.0
                    w_r = self.rho ** days_r if days_r != np.inf else 0.0
                    
                    total_w = w_l + w_r
                    if total_w > 0:
                        daily_res_interp[k] = (w_l * residuals_iter[idx_l] + w_r * residuals_iter[idx_r]) / total_w
                
                # Add the interpolated residuals to the base models predictions
                y_hat_q = df_q_pred['yHat'].values + daily_res_interp
                # Adjust the standard error to account for the AR(1) correlation
                sigma_adj = df_q_pred['SE'].values * np.sqrt(max(0, 1 - self.rho**2))
                
                # Transform back to concentration space and accumulate
                conc_iter = np.exp(y_hat_q + (sigma_adj**2) / 2.0)
                conc += conc_iter
                
                pbar.update(1)

        # Average the niter samples to get the final estimates
        mean_conc = conc / self.niter
        
        df_out = df_q_pred.copy()
        df_out['ConcKalman'] = mean_conc
        
        self._is_fitted = True
        return df_out
