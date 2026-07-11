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
        
        self.surfaces_: np.ndarray | None = None
        self.grid_logq_: np.ndarray | None = None
        self.grid_time_: np.ndarray | None = None
        self.effective_windows_: np.ndarray | None = None
        self._is_fitted: bool = False

    @staticmethod
    def calc_flux(conc: np.ndarray, q: np.ndarray, flow_unit: str = 'cms') -> np.ndarray:
        if flow_unit == 'cms':
            factor = 86.4
        elif flow_unit == 'cfs':
            factor = 2.44657555
        else:
            raise ValueError("flow_unit must be either 'cms' or 'cfs'.")

        return conc * q * factor

    def _validate_data(
        self, df_sample: pd.DataFrame | None = None, df_daily: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        
        df_s = df_sample.copy() if df_sample is not None else None
        df_d = df_daily.copy() if df_daily is not None else None

        for df in [df_s, df_d]:
            if df is not None:
                if not isinstance(df.index, pd.DatetimeIndex):
                    if 'Date' in df.columns:
                        df.set_index('Date', inplace=True)
                        df.index = pd.to_datetime(df.index)
                    else:
                        raise ValueError("DataFrames must have a DatetimeIndex or 'Date' column.")
                if not df.index.is_monotonic_increasing:
                    df.sort_index(inplace=True)

        if df_d is not None:
            if 'LogQ' not in df_d.columns:
                if 'Q' not in df_d.columns:
                    raise ValueError("df_daily must contain 'Q' or 'LogQ'.")
                df_d['LogQ'] = np.log(df_d['Q'].replace(0, np.nan))
                
            if df_s is not None and 'LogQ' not in df_s.columns:
                df_s['LogQ'] = df_d['LogQ'].reindex(df_s.index)
                if df_s['LogQ'].isna().any():
                    raise ValueError("Sample dates missing corresponding discharge in df_daily.")
        else:
            if df_s is not None and 'LogQ' not in df_s.columns:
                raise ValueError("If df_daily is None, df_sample must contain 'LogQ'.")

        # Only validate concentration if a sample dataframe was actually provided
        if df_s is not None:
            if 'Conc' in df_s.columns and 'Uncen' in df_s.columns:
                df_s['ConcHigh'] = df_s['Conc']
                df_s['ConcLow'] = np.where(df_s['Uncen'] == 1, df_s['Conc'], 0.0)
            elif 'ConcHigh' not in df_s.columns or 'Uncen' not in df_s.columns:
                raise ValueError("df_sample must contain 'Conc' and 'Uncen'.")

        for df in [df_s, df_d]:
            if df is not None and 'DecYear' not in df.columns:
                days_in_year = df.index.is_leap_year.map({True: 366.0, False: 365.0})
                df['DecYear'] = df.index.year + (df.index.dayofyear - 0.5) / days_in_year

        return df_s, df_d
