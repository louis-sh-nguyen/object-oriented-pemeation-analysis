from dataclasses import dataclass, field
from typing import Optional

@dataclass
class BaseParameters:
    """
    Experimental conditions
    
    Required Parameters:
    -------------------
    pressure : float
        Applied pressure [bar]
    temperature : float
        Operating temperature [°C]
        
    Optional Parameters:
    -------------------
    thickness : float, optional
        Membrane thickness [cm]
    diameter : float, optional
        Membrane diameter [cm]
    flowrate : float, optional
        Gas flow rate [cm³(STP) s⁻¹]
    """
    pressure: float      # [bar]
    temperature: float   # [°C]
    thickness: Optional[float] = None    # [cm]
    diameter: Optional[float] = None     # [cm]
    flowrate: Optional[float] = None     # [cm³(STP) s⁻¹]

    def validate(self) -> None:
        """Validate parameter values"""
        if self.pressure <= 0:
            raise ValueError("Pressure must be positive")
        if self.temperature < -273.15:
            raise ValueError("Temperature must be above absolute zero")
            
        if self.thickness is not None and self.thickness <= 0:
            raise ValueError("Thickness must be positive")
        if self.diameter is not None and self.diameter <= 0:
            raise ValueError("Diameter must be positive")
        if self.flowrate is not None and self.flowrate <= 0:
            raise ValueError("Flow rate must be positive")

@dataclass
class ModelParameters:
    """Base model parameters"""
    base: BaseParameters
    
    def validate(self) -> None:
        self.base.validate()