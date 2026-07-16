import numpy as np
from scipy.optimize import minimize
from scipy.special import erf
from typing import NamedTuple
from wrtds_py.models.utils import tricube_weight, cyclic_distance
from wrtds_py.base import WindowParams

_LOG_SQRT_2_PI = 0.5 * np.log(2 * np.pi)

class NodeConfig(NamedTuple):
    target_logq: float
    target_time: float
    min_obs: int
    min_uncen: int
    adjust_edges: bool

class NodeResult(NamedTuple):
    y_hat: float
    sigma: float
    conc_hat: float
    final_windows: WindowParams

def _tobit_neg_log_likelihood(x: np.ndarray, model_data: dict) -> float:
    betas = x[:-1]
    sigma = x[-1]
    
    if sigma <= 1e-5:
        return 1e10 
        
    mu = model_data['X'] @ betas

    z = (model_data['y'] - mu) / sigma 
    
    # Gaussian log-PDF for uncensored data
    log_pdf = -0.5 * (z ** 2) - _LOG_SQRT_2_PI - np.log(sigma)
    loglik_uncen = model_data['weights'] * model_data['uncen'] * log_pdf
    
    # Gaussian CDF for left-censored data
    # Because this is a detection limit, the CDF gives the total probability of the distribution up to the detection
    # limit
    cdf = np.clip(0.5 * (1.0 + erf(z / np.sqrt(2.0))), 1e-10, 1.0)
    loglik_cen = model_data['weights'] * (1 - model_data['uncen']) * np.log(cdf)
    
    return -np.sum(loglik_uncen + loglik_cen)

def estimate_surface_node(
    node: NodeConfig,
    sample_data: dict,
    initial_window: WindowParams
) -> NodeResult:
    
    current_windows = initial_window
    
    # Calculate the distance to the nearest edge of the sampe data
    dec_low = np.min(sample_data['time'])
    dec_high = np.max(sample_data['time'])
    dist_time = min(node.target_time - dec_low, dec_high - node.target_time)
    
    # If edge_adjust is True, expand the time window if the target is near the edges of the observations
    # This is to prevent having too few observations on one side of the target time, and many in the other,
    # which can bias the regression.
    if node.adjust_edges and not np.isnan(dist_time):
        if dist_time <= current_windows.time:
            current_windows = WindowParams(
                time=(2 * current_windows.time) - dist_time,
                discharge=current_windows.discharge,
                season=current_windows.season
            )
            
    # Calculate distances between target and sample data for time, logq, and season
    dist_t = node.target_time - sample_data['time']
    dist_q = node.target_logq - sample_data['logq']
    dist_s = cyclic_distance(node.target_time, sample_data['time'])

    # Search for the smallest window that meets the minimum required number of observations and uncensored data
    k = 1
    while True:
        w_t, w_q, w_s = current_windows.as_tuple()
        
        weight_t = tricube_weight(dist_t, w_t)
        weight_q = tricube_weight(dist_q, w_q)
        weight_s = tricube_weight(dist_s, w_s)
        
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
    y_active = sample_data['y'][active_mask]
    uncen = sample_data['uncen'][active_mask]

    # Prepare inputs as the design matrix X    
    X = np.column_stack((
        np.ones(n_active),            # x0
        t_active - node.target_time,  # x1
        q_active - node.target_logq,  # x2
        np.sin(2 * np.pi * t_active), # x3
        np.cos(2 * np.pi * t_active)  # x4
    ))
    
    # Weighted least squares initialization for beta and sigma
    W = np.diag(w_active)
    try:
        init_betas = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y_active
        init_res = y_active - (X @ init_betas)
        init_sigma = max(np.sqrt(np.average(init_res**2, weights=w_active)), 0.05) 
    except np.linalg.LinAlgError:
        init_betas = np.zeros(5)
        init_sigma = 1.0
    
    # Pack the model data into a dict for the likelihood function
    model_data = {'X': X, 'y': y_active, 'uncen': uncen, 'weights': w_active}

    result = minimize(
        _tobit_neg_log_likelihood, 
        x0=np.append(init_betas, init_sigma), 
        args=(model_data,),
        method='L-BFGS-B',
        bounds=[(None, None)] * 5 + [(1e-5, None)]
    )
    
    final_params = result.x if result.success else np.append(init_betas, init_sigma)
    final_betas, final_sigma = final_params[:-1], final_params[-1]
    
    # Use the fitted parameters to predict the log-concentration at the target node
    # Here we only use the intercept and the seasonal terms, since the other terms are zero at the target node
    y_hat = final_betas[0] \
        + final_betas[3] * np.sin(2 * np.pi * node.target_time) \
        + final_betas[4] * np.cos(2 * np.pi * node.target_time)
    
    # Convert log-concentration to concentration using the expected value of a log-normal distribution
    # This accounts for the bias introduced by the log transformation
    conc_hat = np.exp(y_hat + (final_sigma ** 2) / 2.0) 
    
    return NodeResult(y_hat, final_sigma, conc_hat, current_windows)
