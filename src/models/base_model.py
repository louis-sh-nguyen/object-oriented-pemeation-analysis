from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from .parameters import BaseParameters, ModelParameters, TransportParams

class PermeationModel(ABC):
    """
    Abstract base class for permeation models.
    
    Units:
    - Thickness / cm
    - Diameter / cm
    - Flowrate / cm³(STP) s⁻¹
    - Pressure / bar
    - Temperature / °C
    - Diffusion coefficient / cm² s⁻¹
    - Permeability / cm³(STP) cm⁻¹ s⁻¹ bar⁻¹
    - Solubility Coefficient / cm³(STP) cm⁻³ bar⁻¹
    - Equilibrium Concentration / cm³(STP) cm⁻³
    """
    
    def __init__(self, params: ModelParameters):
        """Initialize model with parameters"""
        self.params = params
        self.area = np.pi * (params.base.diameter/2)**2  # [cm²]
        self.results: Dict[str, Any] = {}
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """Validate model parameters"""
        self.params.validate()
    
    @classmethod
    def from_parameter_objects(cls, base_params: BaseParameters, 
                       transport_params: Optional[TransportParams] = None) -> 'PermeationModel':
        """Create model instance from parameters"""
        return cls(ModelParameters(
            base=base_params,
            transport=transport_params or TransportParams()
        ))
    
    @classmethod
    def from_data(cls, data: pd.DataFrame, base_params: BaseParameters) -> Tuple['PermeationModel', pd.DataFrame]:
        """Create model instance and fit to experimental data"""
        model = cls(ModelParameters(base=base_params))
        processed_data = model.fit_to_data(data)
        return model, processed_data
    
    @abstractmethod
    def fit_to_data(self, data: pd.DataFrame) -> None:
        """Fit model to experimental data"""
        pass
    
    @abstractmethod
    def calculate_diffusivity(self, data: pd.DataFrame) -> float:
        """Calculate diffusion coefficient [cm² s⁻¹]"""
        pass
    
    @abstractmethod
    def calculate_permeability(self, data: pd.DataFrame) -> float:
        """Calculate permeability [cm³(STP) cm⁻¹ s⁻¹ bar⁻¹]"""
        pass
    
    @abstractmethod
    def calculate_equilibrium_concentration(self) -> float:
        """Calculate solubility coefficient [cm³(STP) cm⁻³ bar⁻¹]"""
        pass
    
    def get_results(self) -> Dict[str, Any]:
        """Return analysis results"""
        return self.results.copy()