import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

def tricube_weight(distance: float | np.ndarray, half_window: float) -> np.ndarray:
    """Computes the tri-cube weight for a given distance and window boundary."""
    d = np.abs(distance)
    ratio = d / half_window
    return np.where(ratio <= 1.0, (1.0 - ratio**3)**3, 0.0)

def cyclic_distance(time_target: float | np.ndarray, time_sample: np.ndarray) -> np.ndarray:
    """Computes the shortest distance between two dates within a 1-year cycle."""
    frac_target = time_target % 1.0
    frac_sample = time_sample % 1.0
    diff = np.abs(frac_target - frac_sample)
    return np.where(diff > 0.5, 1.0 - diff, diff)

def _tobit_neg_log_likelihood(params: np.ndarray, X: np.ndarray, y_high: np.ndarray, uncen: np.ndarray, weights: np.ndarray) -> float:
    """Computes the negative log-likelihood for the interval-censored weighted regression."""
    beta = params[:-1]
    sigma = params[-1]
    
    if sigma <= 1e-5:
        return 1e10 
        
    mu = X @ beta
    z = (y_high - mu) / sigma
    
    log_pdf = norm.logpdf(z) - np.log(sigma)
    ll_uncen = weights * uncen * log_pdf
    
    cdf = np.clip(norm.cdf(z), 1e-10, 1.0)
    ll_cen = weights * (1 - uncen) * np.log(cdf)
    
    return -np.sum(ll_uncen + ll_cen)

def estimate_surface_node(
    target_logq: float,
    target_time: float,
    sample_time: np.ndarray,
    sample_logq: np.ndarray,
    sample_y_high: np.ndarray,
    sample_uncen: np.ndarray,
    windows: tuple[float, float, float],
    min_obs: int,
    min_uncen: int,
    dec_low: float,
    dec_high: float,
    edge_adjust: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the localized Tobit regression for a SINGLE grid node.
    Arguments reordered to respect the (Q, Time) dimensional hierarchy.
    """
    w_t, w_q, w_s = windows
    
    # 1. Edge Adjustment (Pre-expansion)
    dist_low = target_time - dec_low
    dist_high = dec_high - target_time
    dist_time = min(dist_low, dist_high)
    
    if edge_adjust and not np.isnan(dist_time):
        if dist_time <= w_t:
            w_t = (2 * w_t) - dist_time
            
    # 2. Adaptive Window Expansion Loop
    k = 1
    while True:
        weight_t = tricube_weight(target_time - sample_time, w_t)
        weight_q = tricube_weight(target_logq - sample_logq, w_q)
        weight_s = tricube_weight(cyclic_distance(target_time, sample_time), w_s)
        
        total_weights = weight_t * weight_q * weight_s
        
        active_mask = total_weights > 0
        n_active = np.sum(active_mask)
        n_uncen = np.sum(sample_uncen[active_mask])
        
        if (n_active >= min_obs and n_uncen >= min_uncen) or k > 10000:
            break
            
        w_t *= 1.1
        w_q *= 1.1
        if w_s <= 0.5:
            w_s = min(w_s * 1.1, 0.5)
            
        k += 1

    if k > 10000:
        return np.array([np.nan, np.nan, np.nan]), np.array([w_t, w_q, w_s])

    # 3. Prepare Data for Regression
    w_active = total_weights[active_mask]
    w_active = w_active / (np.sum(w_active) / n_active)
    
    t_active = sample_time[active_mask]
    q_active = sample_logq[active_mask]
    y_high = sample_y_high[active_mask]
    uncen = sample_uncen[active_mask]
    
    N = n_active
    X = np.zeros((N, 5))
    X[:, 0] = 1.0                                       
    X[:, 1] = t_active - target_time                    
    X[:, 2] = q_active - target_logq                    
    X[:, 3] = np.sin(2 * np.pi * t_active)              
    X[:, 4] = np.cos(2 * np.pi * t_active)              
    
    # 4. Tobit Optimization (MLE)
    W = np.diag(w_active)
    try:
        beta_init = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y_high
        res_init = y_high - (X @ beta_init)
        sigma_init = np.sqrt(np.average(res_init**2, weights=w_active))
        sigma_init = max(sigma_init, 0.05) 
    except np.linalg.LinAlgError:
        beta_init = np.zeros(5)
        sigma_init = 1.0
        
    initial_params = np.append(beta_init, sigma_init)
    bounds = [(None, None)] * 5 + [(1e-5, None)] 
    
    result = minimize(
        _tobit_neg_log_likelihood, 
        x0=initial_params, 
        args=(X, y_high, uncen, w_active),
        method='L-BFGS-B',
        bounds=bounds
    )
    
    if not result.success:
        final_beta = beta_init
        final_sigma = sigma_init
    else:
        final_beta = result.x[:-1]
        final_sigma = result.x[-1]
    
    # 5. Final Predictions
    y_hat = final_beta[0] + final_beta[3]*np.sin(2*np.pi*target_time) + final_beta[4]*np.cos(2*np.pi*target_time)
    se = final_sigma
    conc_hat = np.exp(y_hat + (se**2) / 2.0) 
    
    results = np.array([y_hat, se, conc_hat])
    final_windows = np.array([w_t, w_q, w_s])
    
    return results, final_windows
