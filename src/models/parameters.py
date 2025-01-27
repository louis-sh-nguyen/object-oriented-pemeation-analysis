from dataclasses import dataclass
from typing import Optional

@dataclass
class BaseParameters:
    """
    Required experimental parameters
    
    Units:
    - thickness / cm
    - diameter / cm
    - flow_rate / cm³(STP) s⁻¹
    - pressure / bar
    - temperature / °C
    """
    thickness: float
    diameter: float
    flow_rate: float
    pressure: float
    temperature: float = 25.0

    def validate(self) -> None:
        """Validate parameter values"""
        if self.thickness <= 0:
            raise ValueError("Thickness must be positive")
        if self.diameter <= 0:
            raise ValueError("Diameter must be positive")
        if self.flow_rate <= 0:
            raise ValueError("Flow rate must be positive")
        if self.pressure <= 0:
            raise ValueError("Pressure must be positive")
        if self.temperature < -273.15:
            raise ValueError("Temperature must be above absolute zero")

@dataclass
class ModelParameters:
    """
    Complete model parameters including fitted values
    
    Units:
    - diffusivity / cm² s⁻¹
    - permeability / cm³(STP) cm⁻¹ s⁻¹ bar⁻¹
    - solubility_coefficient / cm³(STP) cm⁻³ bar⁻¹
    - solubility / cm³(STP) cm⁻³
    """
    base: BaseParameters
    diffusivity: Optional[float] = None
    permeability: Optional[float] = None
    solubility_coefficient: Optional[float] = None
    solubility: Optional[float] = None

    def validate(self) -> None:
        """Validate all parameters"""
        self.base.validate()
        if self.diffusivity is not None and self.diffusivity <= 0:
            raise ValueError("Diffusivity must be positive")
        if self.permeability is not None and self.permeability <= 0:
            raise ValueError("Permeability must be positive")
        if self.solubility is not None and self.solubility <= 0:
            raise ValueError("Solubility must be positive")