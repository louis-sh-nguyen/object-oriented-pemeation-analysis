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

from .model import TimelagModel
from .plotting import plot_timelag_analysis
from .workflow import time_lag_analysis_workflow

__version__ = '0.1.0'

__all__ = [
    'TimelagModel',
    'plot_timelag_analysis',
    'time_lag_analysis_workflow'
]