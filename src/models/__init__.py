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

from .base_parameters import (
    BaseParameters,
    ModelParameters,
)
from .base_model import PermeationModel
from .single_pressure.constant_diffusivity import (
    TimelagModel,
    TimelagModelParameters,
    TimelagTransportParams,
    plot_timelag_analysis,
    plot_concentration_profile,
    plot_flux_over_time,
    data_fitting_workflow,
)

__version__ = '0.1.0'

__all__ = [
    # Base classes
    'BaseParameters',
    'ModelParameters',
    'TransportParams',
    'PermeationModel',
    
    # Timelag specific
    'TimelagModel',
    'TimelagModelParameters',
    'TimelagTransportParams',
    
    # Analysis functions
    'data_fitting_workflow',
    
    # Plotting functions
    'plot_timelag_analysis',
    'plot_concentration_profile',
    'plot_flux_over_time'
]