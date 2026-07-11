import warnings
import numpy as np
import pandas as pd
from typing import Self
from scipy.interpolate import RegularGridInterpolator

from wrtds_py.base import BaseWRTDS
from wrtds_py.models._estimation import estimate_surface_node, NodeConfig

class WRTDS(BaseWRTDS):
    
    def fit(
        self, 
        df_sample: pd.DataFrame, 
        df_daily: pd.DataFrame | None = None,
        q_min: float | None = None,
        q_max: float | None = None
    ) -> Self:
        
        df_s, df_d = self._validate_data(df_sample, df_daily)
        
        if df_d is not None:
            min_logq = df_d['LogQ'].min() - 0.01 if q_min is None else q_min
            max_logq = df_d['LogQ'].max() + 0.01 if q_max is None else q_max
            grid_min_t, grid_max_t = df_d['DecYear'].min(), df_d['DecYear'].max()
        else:
            if q_min is None or q_max is None:
                raise ValueError("Must provide q_min and q_max if df_daily is None.")
            min_logq, max_logq = q_min, q_max
            grid_min_t, grid_max_t = df_s['DecYear'].min(), df_s['DecYear'].max()

        num_q_nodes = 14 
        grid_logq = np.linspace(min_logq, max_logq, num_q_nodes)
        
        step_time = 1.0 / 16.0 
        num_time_nodes = int(np.ceil((grid_max_t - grid_min_t) / step_time)) + 1
        grid_time = np.linspace(grid_min_t, grid_max_t, num_time_nodes)
        
        self.surfaces_ = np.zeros((num_q_nodes, num_time_nodes, 3))
        self.effective_windows_ = np.zeros((num_q_nodes, num_time_nodes, 3))
        
        sample_data = {
            'time': df_s['DecYear'].values,
            'logq': df_s['LogQ'].values,
            'y_high': np.log(df_s['ConcHigh'].values),
            'uncen': df_s['Uncen'].values
        }
        
        expansion_warnings = 0
        
        for i, target_logq in enumerate(grid_logq):
            for j, target_time in enumerate(grid_time):
                
                self.windows.reset()
                node_config = NodeConfig(
                    target_logq=target_logq,
                    target_time=target_time,
                    min_obs=self.min_num_obs,
                    min_uncen=self.min_num_uncen,
                    edge_adjust=self.edge_adjust
                )
                
                out = estimate_surface_node(node_config, sample_data, self.windows)
                
                self.surfaces_[i, j, :] = out['results']
                self.effective_windows_[i, j, :] = out['windows']
                
                if out['windows'][0] > self.windows.initial_params[0] or out['windows'][1] > self.windows.initial_params[1]:
                    expansion_warnings += 1

        if expansion_warnings > 0:
            warnings.warn(f"Adaptive window expansion triggered for {expansion_warnings} nodes.")

        self.grid_logq_ = grid_logq
        self.grid_time_ = grid_time
        self._is_fitted = True
        self.windows.reset()
        
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("The model is not fitted yet. Call .fit() first.")
        
        _, df_clean = self._validate_data(df_sample=None, df_daily=df)
        
        target_time = np.clip(df_clean['DecYear'].values, self.grid_time_.min(), self.grid_time_.max())
        target_logq = np.clip(df_clean['LogQ'].values, self.grid_logq_.min(), self.grid_logq_.max())
        points = np.column_stack((target_logq, target_time))
        
        interp_yhat = RegularGridInterpolator((self.grid_logq_, self.grid_time_), self.surfaces_[:, :, 0], method='linear')
        interp_se = RegularGridInterpolator((self.grid_logq_, self.grid_time_), self.surfaces_[:, :, 1], method='linear')
        interp_conc = RegularGridInterpolator((self.grid_logq_, self.grid_time_), self.surfaces_[:, :, 2], method='linear')
        
        df_out = df.copy()
        df_out['yHat'] = interp_yhat(points)
        df_out['SE'] = interp_se(points)
        df_out['Conc'] = interp_conc(points)
            
        return df_out
