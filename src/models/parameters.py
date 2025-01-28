from dataclasses import dataclass
from typing import Optional

@dataclass
class BaseParameters:
    """Base parameters for all permeation models"""
    thickness: float  # [cm]
    diameter: float   # [cm]
    flowrate: float  # [cm³(STP) s⁻¹]
    pressure: float  # [bar]
    temperature: float  # [°C]

    def validate(self) -> None:
        """Validate parameter values"""
        if self.thickness <= 0:
            raise ValueError("Thickness must be positive")
        if self.diameter <= 0:
            raise ValueError("Diameter must be positive")
        if self.flowrate <= 0:
            raise ValueError("Flow rate must be positive")
        if self.pressure <= 0:
            raise ValueError("Pressure must be positive")
        if self.temperature < -273.15:
            raise ValueError("Temperature must be above absolute zero")

@dataclass
class ModelParameters:
    """
    Complete model parameters, including base and model parameters (fitted to data or manually specified).
    
    Units:
    - diffusivity / cm² s⁻¹
    - permeability / cm³(STP) cm⁻¹ s⁻¹ bar⁻¹
    - solubility_coefficient / cm³(STP) cm⁻³ bar⁻¹
    - equilibrium_concentration / cm³(STP) cm⁻³
    """
    base: BaseParameters
    diffusivity: Optional[float] = None
    permeability: Optional[float] = None
    solubility_coefficient: Optional[float] = None
    equilibrium_concentration: Optional[float] = None

    def validate(self) -> None:
        """Validate all parameters"""
        self.base.validate()
        if self.diffusivity is not None and self.diffusivity <= 0:
            raise ValueError("Diffusivity must be positive")
        if self.permeability is not None and self.permeability <= 0:
            raise ValueError("Permeability must be positive")
        if self.equilibrium_concentration is not None and self.equilibrium_concentration <= 0:
            raise ValueError("Solubility must be positive")