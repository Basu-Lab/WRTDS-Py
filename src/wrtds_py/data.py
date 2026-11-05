import warnings

import pandas as pd
import numpy as np

# Fraction of the mean discharge added to every day when there are zero-flow days (as in EGRET)
ZERO_FLOW_SHIFT = 0.001

def _log_discharge(q: pd.Series) -> pd.Series:
    """Log of discharge. If any day has zero flow, 0.1% of the mean discharge is added to every
    day before taking the log, following EGRET; 'Q' itself is left unchanged."""
    num_zeros = int((q == 0).sum())
    if num_zeros > 0:
        shift = ZERO_FLOW_SHIFT * q.mean()
        warnings.warn(
            f"There were {num_zeros} zero flow days. LogQ was computed with {shift:.6g} "
            f"({ZERO_FLOW_SHIFT:.1%} of mean discharge) added to every day.",
            UserWarning,
        )
        q = q + shift
    return np.log(q)

def prepare_data(df_wq: pd.DataFrame | None = None, df_q: pd.DataFrame | None = None) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    
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
            df_q['LogQ'] = _log_discharge(df_q['Q'])
        
        # If df_wq doesn't have 'LogQ', map it from df_q using the index
        if df_wq is not None and 'LogQ' not in df_wq.columns:
            df_wq['LogQ'] = df_q['LogQ'].reindex(df_wq.index)
            if df_wq['LogQ'].isna().any():
                raise ValueError("Sample dates missing corresponding discharge in df_q.")
    else:
        if df_wq is not None and 'LogQ' not in df_wq.columns:
            if 'Q' in df_wq.columns:
                df_wq['LogQ'] = _log_discharge(df_wq['Q'])
            else:
                raise ValueError("If df_q is None, df_wq must contain 'Q' or 'LogQ'.")

    if df_wq is not None:
        if 'Conc' in df_wq.columns and 'Uncen' in df_wq.columns:
            df_wq['LogConc'] = np.log(df_wq['Conc'].replace(0, np.nan))
            df_wq['Uncen'] = df_wq['Uncen'].astype(int)
        else:
            raise ValueError("df_wq must contain both 'Conc' and 'Uncen'.")

    return df_wq, df_q