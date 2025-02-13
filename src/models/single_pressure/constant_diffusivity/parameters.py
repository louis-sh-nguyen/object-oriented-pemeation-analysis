from dataclasses import dataclass, field
from typing import Optional
from ...base_parameters import ModelParameters
@dataclass
class TimelagTransportParams:
    """
    Transport and geometric parameters for time-lag model
    
    Geometric Parameters:
    - thickness: Membrane thickness [cm]
    - diameter: Membrane diameter [cm]
    
    Transport Parameters:
    - diffusivity: Diffusion coefficient [cm² s⁻¹]
    - permeability: Permeability coefficient [cm³(STP) cm⁻¹ s⁻¹ bar⁻¹]
    - solubility_coefficient: Solubility coefficient [cm³(STP) cm⁻³ bar⁻¹]
    - equilibrium_concentration: Equilibrium concentration [cm³(STP) cm⁻³]
    - flowrate: Flow rate [cm³(STP) s⁻¹]
    """
    thickness: float                                  # [cm]
    diameter: float                                   # [cm]
    flowrate: Optional[float] = None                 # [cm³(STP) s⁻¹]
    diffusivity: Optional[float] = None              # [cm² s⁻¹]
    permeability: Optional[float] = None             # [cm³(STP) cm⁻¹ s⁻¹ bar⁻¹]
    solubility_coefficient: Optional[float] = None    # [cm³(STP) cm⁻³ bar⁻¹]
    equilibrium_concentration: Optional[float] = None # [cm³(STP) cm⁻³]

    def validate(self) -> None:
        """Validate transport and geometric parameters"""
        # Validate geometric parameters
        if self.thickness <= 0:
            raise ValueError("Thickness must be positive")
        if self.diameter <= 0:
            raise ValueError("Diameter must be positive")
            
        # Validate transport parameters
        if self.flowrate is not None and self.flowrate <= 0:
            raise ValueError("Flow rate must be positive")
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