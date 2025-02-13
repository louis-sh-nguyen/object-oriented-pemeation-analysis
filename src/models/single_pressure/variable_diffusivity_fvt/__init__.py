"""Free Volume Theory (FVT) based variable diffusivity model"""

from ...base_parameters import BaseParameters
from .parameters import FVTModelParameters, FVTTransportParams
from .model import FVTModel

__all__ = [
    'BaseParameters',
    'FVTModelParameters',
    'FVTTransportParams',
    'FVTModel'
]

__version__ = '0.1.0'