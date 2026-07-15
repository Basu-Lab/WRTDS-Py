import numpy as np
from scipy.optimize import minimize
from scipy.special import erf
from dataclasses import dataclass
from typing import NamedTuple
from wrtds_py.models.utils import tricube_weight, cyclic_distance
from wrtds_py.base import WindowParams

# Precompute for speed
LOG_SQRT_2_PI = 0.5 * np.log(2 * np.pi)

@dataclass
class NodeConfig:
    target_logq: float
    target_time: float
    min_obs: int
    min_uncen: int
    edge_adjust: bool

class NodeResult(NamedTuple):
    y_hat: float
    sigma: float
    conc_hat: float
    final_windows: WindowParams

def _tobit_neg_log_likelihood(params_array: np.ndarray, model_data: dict) -> float:
    params_beta = params_array[:-1]
    params_sigma = params_array[-1]
    
    if params_sigma <= 1e-5:
        return 1e10 
        
    mu = model_data['X'] @ params_beta
    # 'y' is now either the observation or the detection limit
    z = (model_data['y'] - mu) / params_sigma 
    
    # Fast Log PDF (Uncensored)
    log_pdf = -0.5 * (z**2) - LOG_SQRT_2_PI - np.log(params_sigma)
    ll_uncen = model_data['weights'] * model_data['uncen'] * log_pdf
    
    # Fast CDF (Left-Censored)
    cdf = np.clip(0.5 * (1.0 + erf(z / np.sqrt(2.0))), 1e-10, 1.0)
    ll_cen = model_data['weights'] * (1 - model_data['uncen']) * np.log(cdf)
    
    return -np.sum(ll_uncen + ll_cen)

def estimate_surface_node(
    node: NodeConfig,
    sample_data: dict,
    initial_windows: WindowParams
) -> NodeResult:
    
    current_windows = initial_windows
    
    dec_low = np.min(sample_data['time'])
    dec_high = np.max(sample_data['time'])
    dist_time = min(node.target_time - dec_low, dec_high - node.target_time)
    
    if node.edge_adjust and not np.isnan(dist_time):
        if dist_time <= current_windows.time:
            current_windows = WindowParams(
                time=(2 * current_windows.time) - dist_time,
                discharge=current_windows.discharge,
                season=current_windows.season
            )
            
    # Calculate static distances ONCE outside the loop for speed
    dist_t_static = node.target_time - sample_data['time']
    dist_q_static = node.target_logq - sample_data['logq']
    dist_s_static = cyclic_distance(node.target_time, sample_data['time'])
            
    k = 1
    while True:
        w_t, w_q, w_s = current_windows.as_tuple()
        
        weight_t = tricube_weight(dist_t_static, w_t)
        weight_q = tricube_weight(dist_q_static, w_q)
        weight_s = tricube_weight(dist_s_static, w_s)
        
        total_weights = weight_t * weight_q * weight_s
        active_mask = total_weights > 0
        n_active = np.sum(active_mask)
        n_uncen = np.sum(sample_data['uncen'][active_mask])
        
        if (n_active >= node.min_obs and n_uncen >= node.min_uncen) or k > 10000:
            break
        
        current_windows = current_windows.expand()
        k += 1

    if k > 10000:
        return NodeResult(np.nan, np.nan, np.nan, current_windows)

    w_active = total_weights[active_mask]
    w_active = w_active / (np.sum(w_active) / n_active)
    
    t_active = sample_data['time'][active_mask]
    q_active = sample_data['logq'][active_mask]
    y_active = sample_data['y'][active_mask] # Updated
    uncen = sample_data['uncen'][active_mask]
    
    X = np.column_stack((
        np.ones(n_active),
        t_active - node.target_time,
        q_active - node.target_logq,
        np.sin(2 * np.pi * t_active),
        np.cos(2 * np.pi * t_active)
    ))
    
    W = np.diag(w_active)
    try:
        beta_init = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y_active
        res_init = y_active - (X @ beta_init)
        sigma_init = max(np.sqrt(np.average(res_init**2, weights=w_active)), 0.05) 
    except np.linalg.LinAlgError:
        beta_init = np.zeros(5)
        sigma_init = 1.0
        
    model_data = {'X': X, 'y': y_active, 'uncen': uncen, 'weights': w_active} # Updated
    
    result = minimize(
        _tobit_neg_log_likelihood, 
        x0=np.append(beta_init, sigma_init), 
        args=(model_data,),
        method='L-BFGS-B',
        bounds=[(None, None)] * 5 + [(1e-5, None)]
    )
    
    final_params = result.x if result.success else np.append(beta_init, sigma_init)
    final_beta, final_sigma = final_params[:-1], final_params[-1]
    
    y_hat = final_beta[0] + final_beta[3]*np.sin(2*np.pi*node.target_time) + final_beta[4]*np.cos(2*np.pi*node.target_time)
    conc_hat = np.exp(y_hat + (final_sigma**2) / 2.0) 
    
    return NodeResult(y_hat, final_sigma, conc_hat, current_windows)
