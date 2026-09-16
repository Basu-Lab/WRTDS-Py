import numpy as np

from wrtds_py.data import prepare_data

class BaseWRTDS:
    def __init__(
        self, 
        window_time: float = 7.0, 
        window_discharge: float = 2.0, 
        window_season: float = 0.5,
        min_num_obs: int = 50, 
        min_num_uncen: int = 30, 
        adjust_edges: bool = True,
        allow_expansion: bool = True,
    ) -> None:
        
        self.window_time = window_time
        self.window_discharge = window_discharge
        self.window_season = window_season
        self.min_num_obs = min_num_obs
        self.min_num_uncen = min_num_uncen
        self.adjust_edges = adjust_edges
        self.allow_expansion = allow_expansion
        
        self.surface_: np.ndarray | None = None
        self.grid_logq_: np.ndarray | None = None
        self.grid_time_: np.ndarray | None = None
        self.effective_windows_: np.ndarray | None = None
        self._is_fitted: bool = False

    def _prepare_data(self, df_wq = None, df_q = None):
        return prepare_data(df_wq, df_q)
