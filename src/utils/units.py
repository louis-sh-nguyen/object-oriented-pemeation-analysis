import numpy as np
from typing import Union, Optional

# Constants
BAR_TO_PA = 1e5  # 1 bar = 100,000 Pa
ZERO_CELSIUS_IN_KELVIN = 273.15
CM3_TO_M3 = 1e-6
BARRER_TO_SI = 3.348e-16  # mol⋅m/(m²⋅s⋅Pa)

def bar_to_pascal(pressure: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert pressure from bar to Pascal"""
    return pressure * BAR_TO_PA

def pascal_to_bar(pressure: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert pressure from Pascal to bar"""
    return pressure / BAR_TO_PA

def celsius_to_kelvin(temp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert temperature from Celsius to Kelvin"""
    return temp + ZERO_CELSIUS_IN_KELVIN

def kelvin_to_celsius(temp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert temperature from Kelvin to Celsius"""
    return temp - ZERO_CELSIUS_IN_KELVIN

def cm3_to_m3(volume: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert volume from cm³ to m³"""
    return volume * CM3_TO_M3

def m3_to_cm3(volume: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert volume from m³ to cm³"""
    return volume / CM3_TO_M3

def barrer_to_si(permeability: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert permeability from Barrer to SI units (mol⋅m/(m²⋅s⋅Pa))"""
    return permeability * BARRER_TO_SI

def si_to_barrer(permeability: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert permeability from SI units to Barrer"""
    return permeability / BARRER_TO_SI

def convert_permeability(
    value: float,
    from_units: str,
    to_units: str
) -> float:
    """
    Convert permeability between different units.
    
    Units supported:
    - 'barrer': 10⁻¹⁰ cm³(STP)⋅cm/(cm²⋅s⋅cmHg)
    - 'si': mol⋅m/(m²⋅s⋅Pa)
    - 'gpu': 10⁻⁶ cm³(STP)/(cm²⋅s⋅cmHg)
    """
    units_dict = {
        'barrer': 1.0,
        'si': BARRER_TO_SI,
        'gpu': 1e-4  # 1 Barrer/cm = 1 GPU
    }
    
    if from_units not in units_dict or to_units not in units_dict:
        raise ValueError(f"Unsupported units. Use: {list(units_dict.keys())}")
    
    # Convert to Barrer first, then to target units
    in_barrer = value / units_dict[from_units]
    return in_barrer * units_dict[to_units]