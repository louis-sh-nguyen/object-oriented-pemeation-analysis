import numpy as np
import pandas as pd
from scipy.stats import linregress
from typing import Optional, Tuple, Dict

def find_stabilisation_time(
    data: pd.DataFrame,
    flux_col: str = 'flux',
    time_col: str = 'time',
    window: int = 100,
    threshold: float = 0.001,
    min_flux_col_value: float = 0.01,
) -> float:
    """Find time when flux stabilizes using moving average analysis."""
    if len(data) < window:
        raise ValueError(f"Data length ({len(data)}) must be >= window size ({window})")
    
    # Filter out initial region where flux is below min_flux_value
    data = data[data[flux_col] > min_flux_col_value].copy()
    
    # Calculate moving statistics
    flux_ma = data[flux_col].rolling(window=window, center=True).mean()
    flux_std = data[flux_col].rolling(window=window, center=True).std()
    
    # Calculate relative changes
    rel_change = abs(flux_ma.diff() / flux_ma)
    rel_std = flux_std / flux_ma
    
    # Find stable region
    # Option 1: Combined criteria (original)
    stable_mask = (rel_change < threshold) & (rel_std < threshold)
    
    if not stable_mask.any():
        print(f"Warning: Flux does not stabilize within threshold {threshold}")
        return float(data.iloc[-1][time_col])
    
    else:
        # Get first stable point
        stab_idx = stable_mask.idxmax()
        return float(data.loc[stab_idx, time_col])

def identify_start_time(
    df: pd.DataFrame, column: str, window: int = 5, threshold: float = 0.001
) -> float:
    """Identify where flux has stabilised using rolling fractional changes.

    Compares the rolling fractional changes of the gradient of a specified
    column with respect to 'time'.

    Args:
        df: Preprocessed DataFrame containing 'time' and the specified column.
        column: Column name to check for stabilisation (e.g., 'cumulative flux / cm^3(STP) cm^-2').
        window: Window size for rolling calculation.
        threshold: Fractional threshold for determining stabilisation.

    Returns:
        Time corresponding to where the specified column has stabilised.

    Raises:
        ValueError: If required columns ('t / s' or the specified column) are missing.
    """
    df = df.copy()

    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    if "time" not in df.columns:
        raise ValueError("Column 'time' does not exist in the DataFrame.")

    df["gradient"] = df[column].diff() / df["time"].diff()
    df["pct_change_mean"] = (
        (df[column].diff() / df["time"].diff())
        .pct_change()
        .abs()
        .rolling(window=window)
        .mean()
    )
    df["pct_change_min"] = (
        (df[column].diff() / df["time"].diff())
        .pct_change()
        .abs()
        .rolling(window=window)
        .min()
    )
    df["pct_change_max"] = (
        (df[column].diff() / df["time"].diff())
        .pct_change()
        .abs()
        .rolling(window=window)
        .max()
    )
    df["pct_change_median"] = (
        (df[column].diff() / df["time"].diff())
        .pct_change()
        .abs()
        .rolling(window=window)
        .median()
    )
    stabilisation_index = df[(df["pct_change_mean"] <= threshold)].index[0]
    stabilisation_time = df.loc[stabilisation_index, "time"]
    return stabilisation_time

def find_time_lag(
    data: pd.DataFrame,
    stabilisation_time: float,
    time_col: str = 'time',
    cumulative_col: str = 'cumulative_flux'
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
