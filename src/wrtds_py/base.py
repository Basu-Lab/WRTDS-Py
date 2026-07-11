import numpy as np
import pandas as pd
from dataclasses import dataclass
@dataclass
class WindowParams:
    time: float
    discharge: float
    season: float

    def __init__(self, time: float, discharge: float, season: float):
        self.time = time
        self.discharge = discharge
        self.season = season
        self.initial_params = (time, discharge, season)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.time, self.discharge, self.season)
    
    def expand(self, factor: float = 1.1, season_cap: float = 0.5):
        self.time *= factor
        self.discharge *= factor
        self.season = min(self.season * factor, season_cap)

    def reset(self):
        self.time, self.discharge, self.season = self.initial_params

class BaseWRTDS:
    def __init__(
        self, 
        window_time: float = 7.0, 
        window_discharge: float = 2.0, 
        window_season: float = 0.5,
        min_num_obs: int = 100, 
        min_num_uncen: int = 50, 
        edge_adjust: bool = True
    ) -> None:
        
        self.windows = WindowParams(window_time, window_discharge, window_season)
        self.min_num_obs = min_num_obs
        self.min_num_uncen = min_num_uncen
        self.edge_adjust = edge_adjust
        
        # Post-fit attributes initialized as None
        self.surfaces_: np.ndarray | None = None
        self.grid_logq_: np.ndarray | None = None
        self.grid_time_: np.ndarray | None = None
        self.effective_windows_: np.ndarray | None = None
        self._is_fitted: bool = False

    def _validate_data(
        self, df_daily: pd.DataFrame, df_sample: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Validates DataFrames and auto-fills missing EGRET structure.
        """
        if not isinstance(df_daily, pd.DataFrame) or not isinstance(df_sample, pd.DataFrame):
            raise TypeError("Inputs must be pandas DataFrames.")

        df_d = df_daily.copy()
        df_s = df_sample.copy()

        # Enforce DatetimeIndex
        for df in [df_d, df_s]:
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'Date' in df.columns:
                    df.set_index('Date', inplace=True)
                    df.index = pd.to_datetime(df.index)
                else:
                    raise ValueError("DataFrames must have a DatetimeIndex or 'Date' column.")
            if not df.index.is_monotonic_increasing:
                df.sort_index(inplace=True)

        # Discharge Handling
        if 'LogQ' not in df_d.columns:
            if 'Q' not in df_d.columns:
                raise ValueError("df_daily must contain 'Q' or 'LogQ'.")
            df_d['LogQ'] = np.log(df_d['Q'].replace(0, np.nan))

        if 'LogQ' not in df_s.columns:
            df_s['LogQ'] = df_d['LogQ'].reindex(df_s.index)
            if df_s['LogQ'].isna().any():
                raise ValueError("Sample dates missing corresponding discharge in df_daily.")

        # Spatial Coordinates (Decimal Year)
        for df in [df_d, df_s]:
            if 'DecYear' not in df.columns:
                days_in_year = df.index.is_leap_year.map({True: 366.0, False: 365.0})
                df['DecYear'] = df.index.year + (df.index.dayofyear - 0.5) / days_in_year

        return df_d, df_s
