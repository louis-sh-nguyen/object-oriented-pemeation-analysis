import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List, Tuple
import os

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

def calculate_flux(
    data: pd.DataFrame,
    flow_rate: float,
    area: float,
    mole_fraction_col: str = 'yCO2',
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
    if not all(col in data.columns for col in [time_col, mole_fraction_col]):
        raise ValueError(f"Required columns {time_col} and/or {mole_fraction_col} missing")
    
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
    df['cumulative flux'] = np.cumsum(df['flux'] * dt)
    
    return df['cumulative flux']

def preprocess_data(
    data: pd.DataFrame,
    thickness: float,
    diameter: float,
    temp_celsius: float,
    flowrate: Optional[float] = None,
    time_col: str = 'time',
    pressure_col: str = 'pressure',
) -> pd.DataFrame:
    """
    Preprocess raw experimental data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Raw experimental data
    thickness : float
        Membrane thickness [cm]
    diameter : float
        Membrane diameter [cm]
    flowrate : float, optional
        Carrier gas flow rate [cm³/s]
    temp_celsius : float
        Temperature [°C]
    time_col : str
        Name of time column
    pressure_col : str
        Name of pressure column
    
    Returns
    -------
    pd.DataFrame
        Processed data with calculated parameters
    """
    # Create copy to avoid modifying original
    df = data.copy()
    
    # Calculate membrane area
    area = np.pi * (diameter/2)**2  # cm²
    
    # Calculate flux if flow rate is provided
    if flowrate is not None:
        df['flux'] = calculate_flux(df, flowrate, area)
        
        # Calculate cumulative flux
        df['cumulative_flux'] = calculate_cumulative_flux(df)
    
    # Add metadata
    df.attrs['thickness'] = thickness
    df.attrs['diameter'] = diameter
    df.attrs['area'] = area
    df.attrs['temperature'] = temp_celsius
    
    return df

def calculate_derived_parameters(
    data: pd.DataFrame,
    pressure_upstream: float,
    molecular_weight: float
) -> Dict[str, float]:
    """
    Calculate derived experimental parameters.
    
    Parameters
    ----------
    data : pd.DataFrame
        Processed experimental data
    pressure_upstream : float
        Upstream pressure [bar]
    molecular_weight : float
        Gas molecular weight [g/mol]
        
    Returns
    -------
    Dict[str, float]
        Dictionary of calculated parameters
    """
    params = {
        'pressure_difference': pressure_upstream,  # Assuming downstream ≈ 0
        'membrane_area': data.attrs.get('area', None),
        'temperature_kelvin': data.attrs.get('temperature', 25.0) + 273.15,
    }
    
    return params