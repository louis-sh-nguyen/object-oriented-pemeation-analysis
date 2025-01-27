import numpy as np
import pandas as pd
from scipy.stats import linregress
from typing import Optional, Tuple, Dict

def find_stabilisation_time(
    data: pd.DataFrame,
    flux_col: str = 'flux',
    time_col: str = 'time',
    window: int = 100,
    threshold: float = 0.01
) -> float:
    """Find time when flux stabilizes using moving average analysis."""
    if len(data) < window:
        raise ValueError(f"Data length ({len(data)}) must be >= window size ({window})")
        
    # Calculate moving statistics
    flux_ma = data[flux_col].rolling(window=window, center=True).mean()
    flux_std = data[flux_col].rolling(window=window, center=True).std()
    
    # Calculate relative changes
    rel_change = abs(flux_ma.diff() / flux_ma)
    rel_std = flux_std / flux_ma
    
    # Find stable region
    stable_mask = (rel_change < threshold) & (rel_std < threshold)
    
    if not stable_mask.any():
        raise ValueError(f"Flux does not stabilize within threshold {threshold}")
    
    # Get first stable point
    stab_idx = stable_mask.idxmax()
    return float(data.loc[stab_idx, time_col])

def find_time_lag(
    data: pd.DataFrame,
    stabilisation_time: float,
    flux_col: str = 'flux',
    time_col: str = 'time',
    cumulative_col: str = 'cumulative flux'
) -> Tuple[float, Dict[str, float]]:
    """Calculate time lag from steady-state portion."""
    # Get steady state data
    steady_state = data[data[time_col] >= stabilisation_time].copy()
    
    if len(steady_state) < 2:
        raise ValueError("Insufficient steady-state data points")
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(
        steady_state[time_col],
        steady_state[cumulative_col]
    )
    
    # Calculate time lag and stats
    time_lag = -intercept / slope
    
    if time_lag < 0:
        raise ValueError("Negative time lag calculated")
    
    stats = {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'std_err': std_err
    }
    
    return time_lag, stats

def validate_time_analysis(
    time_lag: float,
    stats: Dict[str, float],
    min_r_squared: float = 0.99
) -> bool:
    """Validate time lag analysis results."""
    if time_lag <= 0:
        return False
    if stats['r_squared'] < min_r_squared:
        return False
    return True