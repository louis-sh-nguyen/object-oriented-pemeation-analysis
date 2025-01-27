"""
Utility Functions
---------------
Common utilities for permeation analysis:
- Data processing
- Time analysis
- Plotting
- Unit conversions
"""

from .data_processing import load_data, preprocess_data, calculate_cumulative_flux
from .time_analysis import (
    find_stabilisation_time,
    find_time_lag,    
)

from .units import (
    bar_to_pascal,
    pascal_to_bar,
    celsius_to_kelvin,
    kelvin_to_celsius
)

__version__ = '0.1.0'

__all__ = [
    # Data processing
    'load_data',
    'preprocess_data',
    
    # Time analysis
    'find_stabilisation_time',
    'find_time_lag',
    'calculate_cumulative_flux',
    
    # Unit conversions
    'bar_to_pascal',
    'pascal_to_bar',
    'celsius_to_kelvin',
    'kelvin_to_celsius'
]