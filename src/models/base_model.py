from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from .base_parameters import BaseParameters, ModelParameters

class PermeationModel(ABC):
    """
    Abstract base class for permeation models.
    
    Units:
    - Thickness / cm
    - Diameter / cm
    - Flowrate / cm³(STP) s⁻¹
    - Pressure / bar
    - Temperature / °C
    """
    
    def __init__(self, params: ModelParameters):
        """Initialize model with parameters"""
        self.params = params
        self.results: Dict[str, Any] = {}
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """Validate model parameters"""
        self.params.validate()
    
    @classmethod
    @abstractmethod
    def from_parameters(cls,
                       thickness: Optional[float] = None,
                       diameter: Optional[float] = None,
                       flowrate: Optional[float] = None,
                       pressure: Optional[float] = None,
                       temperature: float = None,
                       **kwargs) -> 'PermeationModel':
        """Create model instance from parameters"""
        pass
    
    @abstractmethod
    def fit_to_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit model to experimental data"""
        pass