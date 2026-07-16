import warnings
import numpy as np
import pandas as pd
from typing import Self
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

from wrtds_py.base import BaseWRTDS, WindowParams
from wrtds_py.models._estimation import estimate_surface_node, NodeConfig

class WRTDS(BaseWRTDS):
    
    def fit(
        self, 
        df_wq: pd.DataFrame, 
        df_q: pd.DataFrame | None = None,
        q_min: float | None = None,
        q_max: float | None = None,
        verbose: bool = True
    ) -> Self:
        
        # Validate the input DataFrames
        df_wq_clean, df_q_clean = self._validate_data(df_wq, df_q)
        
        if df_q_clean is not None:
            min_logq = df_q_clean['LogQ'].min() - 0.01 if q_min is None else q_min
            max_logq = df_q_clean['LogQ'].max() + 0.01 if q_max is None else q_max
            grid_min_time, grid_max_time = df_q_clean['DecYear'].min(), df_q_clean['DecYear'].max()
        else:
            if q_min is None or q_max is None:
                raise ValueError("Must provide `q_min` and `q_max` if `df_q` is None.")
            min_logq, max_logq = np.log(q_min), np.log(q_max)
            grid_min_time, grid_max_time = df_wq_clean['DecYear'].min(), df_wq_clean['DecYear'].max()

        num_q_nodes = 14 
        grid_logq = np.linspace(min_logq, max_logq, num_q_nodes)
        
        step_time = 1.0 / 16.0 
        num_time_nodes = int(np.ceil((grid_max_time - grid_min_time) / step_time)) + 1
        grid_time = np.linspace(grid_min_time, grid_max_time, num_time_nodes)
        
        self.surface_ = np.zeros((num_q_nodes, num_time_nodes, 3))
        self.effective_windows_ = np.zeros((num_q_nodes, num_time_nodes, 3))
        
        # Collect 'wq' data into a dict to pass to the node estimation function
        sample_data = {
            'time': df_wq_clean['DecYear'].values,
            'logq': df_wq_clean['LogQ'].values,
            'y': df_wq_clean['LogConc'].values, 
            'uncen': df_wq_clean['Uncen'].values
        }
        
        # Set the initial window to search
        initial_windows = WindowParams(
            time=self.window_time,
            discharge=self.window_discharge,
            season=self.window_season
        )
        
        total_nodes = num_q_nodes * num_time_nodes
        expansion_warnings = 0
        
        # Create a list of nodes to fit
        nodes_to_fit = [
            (i, j, target_logq, target_time)
            for i, target_logq in enumerate(grid_logq)
            for j, target_time in enumerate(grid_time)
        ]

        # Fit each node one by one
        with tqdm(total=total_nodes, disable=not verbose, desc="Fitting WRTDS", ascii=True) as pbar:
            for i, j, target_logq, target_time in nodes_to_fit:
                
                node_config = NodeConfig(
                    target_logq=target_logq,
                    target_time=target_time,
                    min_obs=self.min_num_obs,
                    min_uncen=self.min_num_uncen,
                    adjust_edges=self.adjust_edges
                )
                
                out = estimate_surface_node(node_config, sample_data, initial_windows)
                
                self.surface_[i, j, :] = [out.y_hat, out.sigma, out.conc_hat]
                self.effective_windows_[i, j, :] = out.final_windows.as_tuple()
                
                # Check if the effective window was expanded beyond the initial window
                if (out.final_windows.time > initial_windows.time) or \
                   (out.final_windows.discharge > initial_windows.discharge):
                    expansion_warnings += 1
                    
                pbar.update(1)

        if expansion_warnings > 0:
            warnings.warn(f"Adaptive window expansion triggered for {expansion_warnings} nodes.")

        self.grid_logq_ = grid_logq
        self.grid_time_ = grid_time
        self._is_fitted = True
        
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("The model is not fitted yet. Call .fit() first.")
        
        # Clean df as if it were 'df_q'
        _, df_clean = self._validate_data(df_wq=None, df_q=df)
        
        # Clip the target values to the grid range to prevent extrapolation
        target_time = np.clip(df_clean['DecYear'].values, self.grid_time_.min(), self.grid_time_.max())
        target_logq = np.clip(df_clean['LogQ'].values, self.grid_logq_.min(), self.grid_logq_.max())
        points = np.column_stack((target_logq, target_time))
        
        # Interpolate the fitted surface to get predictions
        interp_yhat = RegularGridInterpolator((self.grid_logq_, self.grid_time_), self.surface_[:, :, 0], method='linear')
        interp_se = RegularGridInterpolator((self.grid_logq_, self.grid_time_), self.surface_[:, :, 1], method='linear')
        interp_conc = RegularGridInterpolator((self.grid_logq_, self.grid_time_), self.surface_[:, :, 2], method='linear')
        
        df_out = df.copy()
        df_out['yHat'] = interp_yhat(points)
        df_out['SE'] = interp_se(points)
        df_out['ConcHat'] = interp_conc(points)
            
        return df_out
    