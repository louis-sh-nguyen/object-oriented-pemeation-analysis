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
    - DT_0: Temperature-dependent diffusivity [cm² s⁻¹]
    - flowrate: Flow rate [cm³(STP) min⁻¹]
    """
    thickness: float                   # [cm]
    diameter: float                    # [cm]
    flowrate: Optional[float] = None   # [cm³(STP) min⁻¹]
    D1_prime: Optional[float] = None  # [adim]
    DT_0: Optional[float] = None      # [cm² s⁻¹]
    U_VprimeW: Optional[float] = None   # [adim]

    def validate(self) -> None:
        """Validate transport and geometric parameters"""
        if self.thickness <= 0:
            raise ValueError("Thickness must be positive")
        if self.diameter <= 0:
            raise ValueError("Diameter must be positive")
        if self.flowrate is not None and self.flowrate <= 0:
            raise ValueError("Flowrate must be positive")
        if self.D1_prime is not None and self.D1_prime <= 0:
            raise ValueError("D1_prime must be positive")
        if self.DT_0 is not None and self.DT_0 <= 0:
            raise ValueError("DT_0 must be positive")
        if self.U_VprimeW is not None and self.U_VprimeW <= 0:
            raise ValueError(" / (V'W) must be positive")

@dataclass
class FVTModelParameters(ModelParameters):
    transport: FVTTransportParams = field(default_factory=FVTTransportParams)

    def validate(self) -> None:
        super().validate()
        self.transport.validate()
