from dataclasses import dataclass, field
from typing import Optional
from ...base_parameters import ModelParameters

@dataclass
class FVTTransportParams:
    """
    Transport and geometric parameters for FVT model
    
    Geometric Parameters:
    - thickness: Membrane thickness [cm]
    - diameter: Membrane diameter [cm]
    
    Transport Parameters:
    - D1_prime: Base diffusivity [cm² s⁻¹]
    - D2_prime: Concentration-dependent diffusivity [cm² s⁻¹ cm³(STP)/cm³]
    - D0_T: Temperature-dependent diffusivity [cm² s⁻¹]
    """
    thickness: float                   # [cm]
    diameter: float                    # [cm]
    D1_prime: Optional[float] = None  # [cm² s⁻¹]
    D2_prime: Optional[float] = None  # [cm² s⁻¹ cm³(STP)/cm³]
    D0_T: Optional[float] = None      # [cm² s⁻¹]

    def validate(self) -> None:
        """Validate transport and geometric parameters"""
        if self.thickness <= 0:
            raise ValueError("Thickness must be positive")
        if self.diameter <= 0:
            raise ValueError("Diameter must be positive")
        if self.D1_prime is not None and self.D1_prime <= 0:
            raise ValueError("D1_prime must be positive")
        if self.D2_prime is not None and self.D2_prime <= 0:
            raise ValueError("D2_prime must be positive")
        if self.D0_T is not None and self.D0_T <= 0:
            raise ValueError("D0_T must be positive")

@dataclass
class FVTModelParameters(ModelParameters):
    transport: FVTTransportParams = field(default_factory=FVTTransportParams)

    def validate(self) -> None:
        super().validate()
        self.transport.validate()
