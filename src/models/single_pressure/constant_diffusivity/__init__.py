"""
Constant Diffusivity Models
--------------------------
Implementation of time-lag analysis for constant diffusivity permeation.

Components:
- TimelagModel: Main model implementation
- plot_timelag_analysis: Visualization functions
- time_lag_analysis_workflow: Analysis workflow

Units:
- Thickness / cm
- Diameter / cm
- Flowrate / cm³(STP) s⁻¹
- Pressure / bar
- Temperature / °C
- Diffusion coefficient / cm² s⁻¹
- Permeability / cm³(STP) cm⁻¹ s⁻¹ bar⁻¹
- Solubility Coefficient / cm³(STP) cm⁻³ bar⁻¹
- Solubility / cm³(STP) cm⁻³
"""

from ...base_parameters import BaseParameters, ModelParameters
from .parameters import TimelagModelParameters, TimelagTransportParams
from .model import TimelagModel
from .plotting import (
    plot_timelag_analysis,
    plot_concentration_profile,
    plot_flux_over_time
)
from .workflow import manual_workflow, data_fitting_workflow

__version__ = '0.1.0'

__all__ = [
    # Parameter classes
    'BaseParameters',
    'ModelParameters',
    'TimelagModelParameters',
    'TimelagTransportParams',
    
    # Model class
    'TimelagModel',
    
    # Analysis workflow
    'manual_workflow',
    'data_fitting_workflow',
    
    # Plotting functions
    'plot_timelag_analysis',
    'plot_concentration_profile',
    'plot_flux_over_time'
]