from dataclasses import dataclass, field
from typing import Optional

@dataclass
class BaseParameters:
    """
    Basic parameters for permeation experiments
    
    Required Parameters:
    -------------------
    pressure : float
        Applied gas pressure [bar]
    temperature : float
        Operating temperature [°C]
    """
    pressure: Optional[float] = None      # [bar]
    temperature: Optional[float] = None   # [°C]

    def validate(self) -> None:
        """Validate parameter values"""
        if self.pressure is not None and self.pressure <= 0:
            raise ValueError("Pressure must be positive")
        if self.temperature is not None and self.temperature < -273.15:
            raise ValueError("Temperature must be above absolute zero")

@dataclass
class ModelParameters:
    """Base model parameters"""
    base: BaseParameters
    
    def validate(self) -> None:
        self.base.validate()