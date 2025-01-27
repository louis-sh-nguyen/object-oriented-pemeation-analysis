"""
Single Pressure Step Models
--------------------------
Models for analyzing single pressure step permeation experiments.

Models:
- TimelagModel: Constant diffusivity analysis
- VariableConcentrationModel: Concentration-dependent diffusivity
"""

from .constant_diffusivity.model import TimelagModel
from .constant_diffusivity.plotting import plot_timelag_analysis
from .constant_diffusivity.workflow import time_lag_analysis_workflow

# from .variable_concentration.model import VariableConcentrationModel
# from .variable_concentration.plotting import plot_variable_concentration_analysis
# from .variable_concentration.workflow import variable_concentration_analysis_workflow

__version__ = '0.1.0'

__all__ = [
    'TimelagModel',
    'plot_timelag_analysis',
    'time_lag_analysis_workflow',
    'VariableConcentrationModel',
    'plot_variable_concentration_analysis',
    'variable_concentration_analysis_workflow'
]