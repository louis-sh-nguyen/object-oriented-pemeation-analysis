import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List, Tuple
import os
from .time_analysis import find_stabilisation_time

def validate_columns(data: pd.DataFrame, required_cols: List[str]) -> bool:
    """
    Validate if DataFrame contains required columns.
    """
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return True

def load_data(
    file_path: str,
    time_col: str = 'time',
    pressure_col: str = 'pressure',
    required_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load experimental data from file.
    
    Parameters
    ----------
    file_path : str
        Path to data file (.xlsx or .csv)
    time_col : str
        Name of time column
    pressure_col : str
        Name of pressure column
    required_cols : List[str], optional
        List of required columns
        
    Returns
    -------
    pd.DataFrame
        Loaded and validated data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
        
    # Load data based on file extension
    if file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Use .xlsx or .csv")
    
    # Validate columns if specified
    if required_cols:
        validate_columns(data, required_cols)
    
    # Ensure time starts at 0
    if time_col in data.columns:
        data[time_col] = data[time_col] - data[time_col].min()
    
    return data

def correct_baseline(df: pd.DataFrame, col: str = 'yCO2', n: int = 10) -> pd.DataFrame:
    """
    Correct the baseline of the raw data.

    Parameters:
    df (pd.DataFrame): Raw data.
    col (str): Name of the column to correct.
    baseline (float): Baseline value to subtract from the column.

    Returns:
    pd.DataFrame: Baseline-corrected data.
    """
    df = df.copy()
    baseline = df.iloc[:n, :][col].mean()
    df['yCO2_bl'] = df[col] - baseline
    return df['yCO2_bl']

def calculate_flux(
    data: pd.DataFrame,
    flow_rate: float,
    area: float,
    mole_fraction_col: str = 'yCO2_bl',
    time_col: str = 'time'
) -> pd.DataFrame:
    """
    Calculate flux from mole fraction data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Raw data with time and mole fraction columns
    flow_rate : float
        Carrier gas flow rate [cm³(STP)/s]
    area : float
        Membrane area [cm²]
    mole_fraction_col : str
        Name of mole fraction column
    time_col : str
        Name of time column
        
    Returns
    -------
    pd.DataFrame
        Data with added 'flux' column [cm³(STP)/(cm²⋅s)]
    """
    # Validate inputs
    required_cols = [time_col, mole_fraction_col]
    validate_columns(data, required_cols)
    
    if flow_rate <= 0 or area <= 0:
        raise ValueError("Flow rate and area must be positive")
    
    # Create copy to avoid modifying original
    df = data.copy()
    
    # Calculate flux
    df['flux'] = df[mole_fraction_col] * flow_rate / area
    
    return df['flux']

def calculate_cumulative_flux(
    data: pd.DataFrame,
    flux_col: str = 'flux',
    time_col: str = 'time'
) -> pd.Series:
    """Calculate cumulative flux using trapezoidal integration."""
    if len(data) < 2:
        raise ValueError("At least two data points required")
        
    df = data.copy()
    
    # Calculate time intervals
    dt = np.diff(df[time_col], prepend=0)
    df['cumulative flux'] = np.cumsum(df[flux_col] * dt)
    
    return df['cumulative flux']

def preprocess_data(data: pd.DataFrame,
                    thickness: float,
                    diameter: float,
                    flowrate: float,
                    temperature: float,
                    truncate_at_stabilisation: bool = False,
                    stabilisation_threshold: float = None) -> pd.DataFrame:
    """
    Preprocess experimental permeation data for analysis
    
    Parameters
    ----------
    data : pd.DataFrame
        Raw experimental data containing columns:
        - 'time': Time values [s]
    thickness : float
        Membrane thickness [cm]
    diameter : float
        Membrane diameter [cm]
    flowrate : float
        Volumetric flow rate [cm³(STP) min⁻¹]
    temp_celsius : float
        Temperature [°C]
    truncate_at_stabilisation : bool, optional
        Whether to truncate data at stabilisation time (default: False)
        If True, adds 'stabilisation_time' to DataFrame.attrs
    threshold : float, optional
        Threshold value for stabilisation time (default: None)
    
    Returns
    -------
    pd.DataFrame
        Processed data with columns:
        - 'Time(s)': Time values [s]
        - 'Flux(cm³(STP)/cm².s)': Flux values [cm³(STP) cm⁻² s⁻¹]
        - 'cumulative_flux': Cumulative flux [cm³(STP) cm⁻²]
        
    Notes
    -----
    - If flux column not present, calculates it from flowrate divided by membrane area
    - Cumulative flux is calculated by time integration of flux
    - If truncate_at_stabilisation=True, only returns data up to stabilisation time
    """
    # Create copy to avoid modifying original
    df = data.copy()
    
    # Calculate membrane area
    area = np.pi * (diameter/2)**2  # [cm²]
    
    # Baseline correction
    df['yCO2_bl'] = correct_baseline(df)
    
    # Flux
    df['flux'] = calculate_flux(df, flowrate, area)
    
    # Calculate cumulative flux
    df['cumulative_flux'] = calculate_cumulative_flux(df)
    
    # Noralised flux
    # Calculate 10-period rolling average and get its maximum value
    df['rolling_avg'] = df['flux'].rolling(window=10, center=True).mean()
    rolling_avg_max = df['flux'].rolling(window=10, center=True).mean().max()
    # rolling_avg = df['flux'].rolling(window=20, center=True).mean()
    # Damping factor
    # damping_factor = 0.1
    # Apply exponential smoothing to damp out fluctuations
    # df['smoothed_flux'] = rolling_avg.ewm(alpha=damping_factor).mean()
    # rolling_avg_max = df['smoothed_flux'].max()
    # Normalise flux by dividing by the maximum rolling average
    df['normalised_flux'] = df['flux'] / rolling_avg_max
    
    # Convert barg to bar for pressure
    df['pressure'] = df['pressure'] + 1  # Convert from barg to bar
    
    # Add metadata
    df.attrs['thickness'] = thickness
    df.attrs['diameter'] = diameter
    df.attrs['area'] = area
    df.attrs['temperature'] = temperature
    
    # Find stabilisation time and truncate if requested
    if truncate_at_stabilisation:
        stab_time = find_stabilisation_time(df, flux_col='normalised_flux', threshold=stabilisation_threshold)
        df = df[df['time'] <= stab_time].copy()
        
        # Add stabilisation time to metadata
        df.attrs['stabilisation_time'] = stab_time
    
    # Columns to retain
    columns_to_keep = ['time', 'flux', 'cumulative_flux', 'normalised_flux', 'yCO2_bl', 'pressure', 'temperature']
    
    return df[columns_to_keep]