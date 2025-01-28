from dataclasses import dataclass, field
from typing import Optional

@dataclass
class BaseParameters:
    """Experimental conditions"""
    thickness: float      # [cm]
    diameter: float      # [cm]
    flowrate: float     # [cm³(STP) s⁻¹]
    pressure: float     # [bar]
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
class TransportParams:
    """Transport parameters for permeation models"""
    diffusivity: Optional[float] = None                # [cm² s⁻¹]
    permeability: Optional[float] = None               # [cm³(STP) cm⁻¹ s⁻¹ bar⁻¹]
    solubility_coefficient: Optional[float] = None     # [cm³(STP) cm⁻³ bar⁻¹]
    equilibrium_concentration: Optional[float] = None   # [cm³(STP) cm⁻³]

    def validate(self) -> None:
        """Validate transport parameters"""
        if self.diffusivity is not None and self.diffusivity <= 0:
            raise ValueError("Diffusivity must be positive")
        if self.permeability is not None and self.permeability <= 0:
            raise ValueError("Permeability must be positive")
        if self.solubility_coefficient is not None and self.solubility_coefficient <= 0:
            raise ValueError("Solubility coefficient must be positive")
        if self.equilibrium_concentration is not None and self.equilibrium_concentration <= 0:
            raise ValueError("Equilibrium concentration must be positive")

@dataclass
class ModelParameters:
    """Complete model parameters"""
    base: BaseParameters
    transport: TransportParams = field(default_factory=TransportParams) # field to allow default parameters in TransportParams

    def validate(self) -> None:
        """Validate all parameters"""
        self.base.validate()
        self.transport.validate()