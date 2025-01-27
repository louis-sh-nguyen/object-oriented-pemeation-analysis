"""
Permeation Analysis Models
-------------------------
Models for analyzing gas permeation data through polymeric membranes.

Available Models:
- TimelagModel: Constant diffusivity analysis
- VariableConcentrationModel: Concentration-dependent diffusivity
- MultiStepModel: Multi-pressure step analysis

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

from .parameters import BaseParameters, ModelParameters
from .base_model import PermeationModel
from .single_pressure.constant_diffusivity.model import TimelagModel
# from .single_pressure.variable_concentration.model import VariableConcentrationModel
# from .multi_pressure.model import MultiStepModel

__version__ = '0.1.0'

__all__ = [
    'BaseParameters',
    'ModelParameters',
    'PermeationModel',
    'TimelagModel',
    'VariableConcentrationModel',
    'MultiStepModel'
]