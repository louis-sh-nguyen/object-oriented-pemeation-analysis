from dataclasses import dataclass, field
from typing import Optional
from ...base_parameters import ModelParameters

@dataclass
class TimelagTransportParams:
    """Transport parameters for time-lag model"""
    diffusivity: Optional[float] = None
    permeability: Optional[float] = None
    solubility_coefficient: Optional[float] = None
    equilibrium_concentration: Optional[float] = None

    def validate(self) -> None:
        if self.diffusivity is not None and self.diffusivity <= 0:
            raise ValueError("Diffusivity must be positive")
        if self.permeability is not None and self.permeability <= 0:
            raise ValueError("Permeability must be positive")
        if self.solubility_coefficient is not None and self.solubility_coefficient <= 0:
            raise ValueError("Solubility coefficient must be positive")
        if self.equilibrium_concentration is not None and self.equilibrium_concentration <= 0:
            raise ValueError("Equilibrium concentration must be positive")

@dataclass
class TimelagModelParameters(ModelParameters):
    transport: TimelagTransportParams = field(default_factory=TimelagTransportParams)

    def validate(self) -> None:
        super().validate()
        self.transport.validate()