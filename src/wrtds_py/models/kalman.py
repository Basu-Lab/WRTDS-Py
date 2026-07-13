import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import truncnorm

from wrtds_py.base import BaseWRTDS
from wrtds_py.models.classic import WRTDS

class WRTDSK:
    def __init__(self, base_model: WRTDS, rho: float = 0.9, niter: int = 100, seed: int | None = None):
        if not getattr(base_model, '_is_fitted', False):
            raise ValueError("The base WRTDS model must be fitted before running Kalman.")
        
        self.base_model = base_model
        self.rho = rho
        self.niter = niter
        self.seed = seed
        self._is_fitted = False

    def filter(self, df_sample: pd.DataFrame, df_daily: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        if self.seed is not None:
            np.random.seed(self.seed)

        # 1. Get base predictions for both dataframes
        df_s_pred = self.base_model.predict(df_sample)
        df_d_pred = self.base_model.predict(df_daily)

        # 2. Prepare arrays for the AR(1) math
        y_obs = np.log(df_s_pred['ConcHigh'].values)
        y_hat_s = df_s_pred['yHat'].values
        se_s = df_s_pred['SE'].values
        uncen = df_s_pred['Uncen'].values
        
        t_sample = df_s_pred['DecYear'].values
        t_daily = df_d_pred['DecYear'].values
        
        n_daily = len(df_d_pred)
        n_sample = len(df_s_pred)
        
        sum_conc = np.zeros(n_daily)
        
        # 3. Monte Carlo Iterations
        with tqdm(total=self.niter, disable=not verbose, desc="WRTDS-K Monte Carlo") as pbar:
            for _ in range(self.niter):
                # a) Generate residuals for this iteration
                residuals_iter = np.zeros(n_sample)
                
                for i in range(n_sample):
                    if uncen[i] == 1:
                        # Exact residual for uncensored data
                        residuals_iter[i] = y_obs[i] - y_hat_s[i]
                    else:
                        # Draw from truncated normal for left-censored data
                        a = -np.inf
                        b = (y_obs[i] - y_hat_s[i]) / se_s[i]
                        draw = truncnorm.rvs(a, b, loc=0, scale=se_s[i])
                        residuals_iter[i] = draw * se_s[i]
                
                # b) Interpolate residuals for every daily prediction (Simplified Bridge)
                daily_res_interp = np.zeros(n_daily)
                
                # Vectorized search for nearest left bounds
                left_idx = np.searchsorted(t_sample, t_daily, side='right') - 1
                right_idx = np.clip(left_idx + 1, 0, n_sample - 1)
                
                for k in range(n_daily):
                    idx_l, idx_r = left_idx[k], right_idx[k]
                    if idx_l < 0:
                        idx_l = 0 # Handle edge case before first sample
                    
                    days_l = (t_daily[k] - t_sample[idx_l]) * 365.25
                    days_r = (t_sample[idx_r] - t_daily[k]) * 365.25
                    
                    if days_l < 0:
                        days_l = np.inf
                    if days_r < 0:
                        days_r = np.inf
                    
                    # Calculate exponential decay weights
                    w_l = self.rho ** days_l if days_l != np.inf else 0.0
                    w_r = self.rho ** days_r if days_r != np.inf else 0.0
                    
                    total_w = w_l + w_r
                    if total_w > 0:
                        daily_res_interp[k] = (w_l * residuals_iter[idx_l] + w_r * residuals_iter[idx_r]) / total_w
                
                # c) Recombine and correct standard error variance
                y_hat_d = df_d_pred['yHat'].values + daily_res_interp
                sigma_adj = df_d_pred['SE'].values * np.sqrt(max(0, 1 - self.rho**2))
                
                conc_iter = np.exp(y_hat_d + (sigma_adj**2) / 2.0)
                sum_conc += conc_iter
                
                pbar.update(1)

        # 4. Final Averaging
        mean_conc = sum_conc / self.niter
        
        df_out = df_d_pred.copy()
        df_out['GenConc'] = mean_conc
        df_out['GenFlux'] = BaseWRTDS.calc_flux(mean_conc, df_out['Q'].values)
        
        self._is_fitted = True
        return df_out
