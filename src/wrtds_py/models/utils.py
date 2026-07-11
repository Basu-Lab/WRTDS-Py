import numpy as np

def tricube_weight(distance: float | np.ndarray, half_window: float) -> np.ndarray:
    d = np.abs(distance)
    ratio = d / half_window
    return np.where(ratio <= 1.0, (1.0 - ratio**3)**3, 0.0)

def cyclic_distance(time_target: float | np.ndarray, time_sample: np.ndarray) -> np.ndarray:
    frac_target = time_target % 1.0
    frac_sample = time_sample % 1.0
    diff = np.abs(frac_target - frac_sample)
    return np.where(diff > 0.5, 1.0 - diff, diff)
