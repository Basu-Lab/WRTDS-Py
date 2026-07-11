import numpy as np
import pandas as pd
from typing import Self
from scipy.interpolate import RegularGridInterpolator

from ..base import BaseWRTDS
from ._estimation import estimate_surface_node

class WRTDS(BaseWRTDS):
    """
    The classic WRTDS implementation.
    Inherits data validation and parameters from BaseWRTDS.
    """
    
    def fit(self, df_daily: pd.DataFrame, df_sample: pd.DataFrame) -> Self:
        """
        Fits the local WLS regressions to build the surfaces array.
        """
        df_d, df_s = self._validate_data(df_daily, df_sample)
        
        # 1. Grid Boundaries 
        min_logq, max_logq = df_d['LogQ'].min() - 0.01, df_d['LogQ'].max() + 0.01
        grid_min_t, grid_max_t = df_d['DecYear'].min(), df_d['DecYear'].max()
        
        # 2. Edge Adjustment Boundaries 
        dec_low = df_s['DecYear'].min()
        dec_high = df_s['DecYear'].max()
        
        # 3. Generate Target Coordinates
        num_q_nodes = 14 
        grid_logq = np.linspace(min_logq, max_logq, num_q_nodes)
        
        step_time = 1.0 / 16.0 
        num_time_nodes = int(np.ceil((grid_max_t - grid_min_t) / step_time)) + 1
        grid_time = np.linspace(grid_min_t, grid_max_t, num_time_nodes)
        
        # 4. Initialize Output Grids (Dimension order: 14 Discharge x N Time x 3 Outputs)
        self.surfaces_ = np.zeros((num_q_nodes, num_time_nodes, 3))
        self.effective_windows_ = np.zeros((num_q_nodes, num_time_nodes, 3))
        
        # 5. Extract Fast NumPy Arrays 
        sample_time = df_s['DecYear'].values
        sample_logq = df_s['LogQ'].values
        sample_y_high = np.log(df_s['ConcHigh'].values) # Concentration is fitted in log-space
        sample_uncen = df_s['Uncen'].values
        
        default_windows = self.windows# .copy()  # Create a copy to avoid mutating the original during expansion
        expansion_warnings = 0
        
        # 6. Populate Grid (Outer loop: Discharge, Inner loop: Time)
        for i, target_logq in enumerate(grid_logq):
            for j, target_time in enumerate(grid_time):
                
                results, final_windows = estimate_surface_node(
                    target_logq=target_logq,
                    target_time=target_time,
                    sample_time=sample_time,
                    sample_logq=sample_logq,
                    sample_y_high=sample_y_high,
                    sample_uncen=sample_uncen,
                    windows=default_windows,
                    min_obs=self.min_num_obs,
                    min_uncen=self.min_num_uncen,
                    dec_low=dec_low,
                    dec_high=dec_high,
                    edge_adjust=self.edge_adjust
                )
                
                self.surfaces_[i, j, :] = results
                self.effective_windows_[i, j, :] = final_windows
                
                if final_windows[0] > self.windows.time or final_windows[1] > self.windows.discharge:
                    expansion_warnings += 1

        if expansion_warnings > 0:
            import warnings
            warnings.warn(
                f"Adaptive window expansion triggered for {expansion_warnings} nodes. "
                "Check `effective_windows_` for details."
            )

        self.grid_logq_ = grid_logq
        self.grid_time_ = grid_time
        self._is_fitted = True
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts yHat, Standard Error, and Daily Concentration for a given dataset
        by interpolating from the fitted WRTDS surfaces grid.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The input dataframe (typically your continuous daily discharge record).
            Must contain 'Q' (or 'LogQ') and a DatetimeIndex or 'Date' column.
            
        Returns:
        --------
        pd.DataFrame
            A copy of the input dataframe with 'yHat', 'SE', and 'ConcDay' appended.
        """
        if not self._is_fitted:
            raise RuntimeError("The model is not fitted yet. Call .fit() first.")
        
        # 1. Prepare the input data (Reuse our validation logic cleanly)
        # We pass a dummy sample dataframe just to satisfy the method signature,
        # but we only care about the cleaned daily dataframe returned.
        df_clean, _ = self._validate_data(df_daily=df, df_sample=df.iloc[:2])
        
        # 2. Extract coordinates to predict
        target_time = df_clean['DecYear'].values
        target_logq = df_clean['LogQ'].values
        
        # Safety Check: Clip extreme values to the grid boundaries.
        # WRTDS extrapolates flatly at the very edges to prevent wild predictions.
        target_time = np.clip(target_time, self.grid_time_.min(), self.grid_time_.max())
        target_logq = np.clip(target_logq, self.grid_logq_.min(), self.grid_logq_.max())
        
        # Stack into the (N, 2) shape required by Scipy, matching our (Q, Time) grid order
        points = np.column_stack((target_logq, target_time))
        
        # 3. Build the Interpolators
        # We create a separate interpolator for each of the 3 layers in surfaces_
        interp_yhat = RegularGridInterpolator(
            (self.grid_logq_, self.grid_time_), 
            self.surfaces_[:, :, 0], 
            method='linear'
        )
        interp_se = RegularGridInterpolator(
            (self.grid_logq_, self.grid_time_), 
            self.surfaces_[:, :, 1], 
            method='linear'
        )
        interp_conc = RegularGridInterpolator(
            (self.grid_logq_, self.grid_time_), 
            self.surfaces_[:, :, 2], 
            method='linear'
        )
        
        # 4. Execute Predictions
        df_out = df.copy()
        df_out['yHat'] = interp_yhat(points)
        df_out['SE'] = interp_se(points)
        df_out['ConcDay'] = interp_conc(points)
        
        # Calculate daily flux (Mass Load)
        # EGRET standard flux calculation: Conc (mg/L) * Q (cfs) * 2.44657555 = Flux (kg/day)
        # Assuming Q is in standard USGS cubic feet per second (cfs).
        conversion_factor = 2.44657555
        if 'Q' in df_clean.columns:
            df_out['FluxDay'] = df_out['ConcDay'] * df_clean['Q'] * conversion_factor
            
        return df_out