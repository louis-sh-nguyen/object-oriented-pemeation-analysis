"""
Permeation Analysis Package
--------------------------
Tools for analyzing gas permeation experiments using time-lag method.

Components:
- Single pressure analysis (constant & variable diffusivity)
- Multi-pressure step analysis
- Data processing utilities
- Visualization tools

Units:
- Thickness: cm
- Diameter: cm
- Flow rate: cm³(STP)/s
- Pressure: bar
- Temperature: K or °C
- Diffusion coefficient: cm²/s
- Permeability: cm³(STP)⋅cm/(cm²⋅s⋅bar)
- Solubility: cm³(STP)/(cm³⋅bar)
"""

from .models.single_pressure.constant_diffusivity import (
    TimelagModel,
    data_fitting_workflow
)

from .utils.data_processing import (
    load_data,
    preprocess_data,
    calculate_flux,
    calculate_cumulative_flux
)

from .utils.time_analysis import (
    find_stabilisation_time,
    find_time_lag,    
)

__version__ = '0.1.0'

__all__ = [
    # Models
    'TimelagModel',
    'data_fitting_workflow',
    
    # Data processing
    'load_data',
    'preprocess_data',
    'calculate_flux',
    
    # Time analysis
    'find_stabilisation_time',
    'find_time_lag', 
    'calculate_cumulative_flux'
]