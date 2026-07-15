import numpy as np
import pandas as pd
from dataclasses import dataclass

_CMS_TO_KG_PER_DAY = 86.4
_CFS_TO_KG_PER_DAY = 2.44657555

@dataclass(frozen=True)
class WindowParams:
    time: float
    discharge: float
    season: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.time, self.discharge, self.season)
    
    def expand(self, factor: float = 1.1, season_cap: float = 0.5) -> 'WindowParams':
        return WindowParams(
            time=self.time * factor,
            discharge=self.discharge * factor,
            season=min(self.season * factor, season_cap)
        )

class BaseWRTDS:
    def __init__(
        self, 
        window_time: float = 7.0, 
        window_discharge: float = 2.0, 
        window_season: float = 0.5,
        min_num_obs: int = 50, 
        min_num_uncen: int = 30, 
        edge_adjust: bool = True
    ) -> None:
        
        self.window_time = window_time
        self.window_discharge = window_discharge
        self.window_season = window_season
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
        match flow_unit:
            case 'cms':
                factor = _CMS_TO_KG_PER_DAY
            case 'cfs':
                factor = _CFS_TO_KG_PER_DAY
            case _:
                raise ValueError("flow_unit must be either 'cms' or 'cfs'.")

        return conc * q * factor

    def _validate_data(
        self, df_wq: pd.DataFrame | None = None, df_q: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        
        df_wq = df_wq.copy() if df_wq is not None else None
        df_q = df_q.copy() if df_q is not None else None

        def _set_index(df: pd.DataFrame | None) -> pd.DataFrame | None:
            if df is None:
                return None
            
            # Check if the index is already a DatetimeIndex; if not, attempt to set it
            if not isinstance(df.index, pd.DatetimeIndex):
                valid_index_cols = {'date', 'datetime', 'time', 'timestamp'}
                cols_lower = {col.lower() for col in df.columns}
                index_col = valid_index_cols.intersection(cols_lower) # match columns
                if index_col:
                    date_col = next(iter(index_col))
                    df.index = pd.to_datetime(df[date_col])
            
            if not df.index.is_monotonic_increasing:
                df = df.sort_index()
            
            if 'DecYear' not in df.columns:
                days_in_year = np.where(df.index.is_leap_year, 366.0, 365.0)
                df['DecYear'] = df.index.year + (df.index.dayofyear - 0.5) / days_in_year
            
            return df

        df_wq = _set_index(df_wq)
        df_q = _set_index(df_q)

        if df_q is not None:
            if 'LogQ' not in df_q.columns:
                if 'Q' not in df_q.columns:
                    raise ValueError("df_q must contain 'Q' or 'LogQ'.")
                df_q['LogQ'] = np.log(df_q['Q'].replace(0, np.nan))
            
            # If df_wq doesn't have 'LogQ', map it from df_q using the index
            if df_wq is not None and 'LogQ' not in df_wq.columns:
                df_wq['LogQ'] = df_q['LogQ'].reindex(df_wq.index)
                if df_wq['LogQ'].isna().any():
                    raise ValueError("Sample dates missing corresponding discharge in df_q.")
        else:
            if df_wq is not None and 'LogQ' not in df_wq.columns:
                raise ValueError("If df_q is None, df_wq must contain 'LogQ'.")

        if df_wq is not None:
            if 'Conc' in df_wq.columns and 'Uncen' in df_wq.columns:
                df_wq['LogConc'] = np.log(df_wq['Conc'].replace(0, np.nan))
                df_wq['Uncen'] = df_wq['Uncen'].astype(int)
            elif 'ConcHigh' not in df_wq.columns or 'Uncen' not in df_wq.columns:
                raise ValueError("df_wq must contain both 'Conc' and 'Uncen'.")

        return df_wq, df_q
