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
    pressure: float      # [bar]
    temperature: float   # [°C]

    def validate(self) -> None:
        """Validate parameter values"""
        if self.pressure <= 0:
            raise ValueError("Pressure must be positive")
        if self.temperature < -273.15:
            raise ValueError("Temperature must be above absolute zero")

@dataclass
class ModelParameters:
    """Base model parameters"""
    base: BaseParameters
    
    def validate(self) -> None:
        self.base.validate()