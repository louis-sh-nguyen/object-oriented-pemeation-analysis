import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from ...base_model import PermeationModel
from ...base_parameters import BaseParameters
from .parameters import TimelagModelParameters, TimelagTransportParams
from ....utils.time_analysis import find_stabilisation_time, find_time_lag
from ....utils.data_processing import preprocess_data

class TimelagModel(PermeationModel):
    """Time-lag analysis model for constant diffusivity permeation"""
    
    def __init__(self, params: TimelagModelParameters):
        super().__init__(params)
        self.results: Dict[str, Any] = {}
    
    @classmethod
    def from_parameters(cls,
                       thickness: float,
                       diameter: float,
                       flowrate: float,
                       pressure: float,
                       temperature: float,
                       diffusivity: Optional[float] = None,
                       equilibrium_concentration: Optional[float] = None) -> 'TimelagModel':
        """Create model instance from parameters"""
        base_params = BaseParameters(
            thickness=thickness,
            diameter=diameter,
            flowrate=flowrate,
            pressure=pressure,
            temperature=temperature
        )
        
        transport_params = TimelagTransportParams(
            diffusivity=diffusivity,
            equilibrium_concentration=equilibrium_concentration
        )
        
        return cls(TimelagModelParameters(base=base_params, transport=transport_params))

    def fit_to_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit model to experimental data"""
        processed_data = preprocess_data(
            data,
            thickness=self.params.base.thickness,
            diameter=self.params.base.diameter,
            flowrate=self.params.base.flowrate,
            temp_celsius=self.params.base.temperature
        )
        
        self.calculate_diffusivity(processed_data)
        self.calculate_permeability(processed_data)
        self.calculate_solubility_coefficient()
        self.calculate_equilibrium_concentration()
        
        return processed_data

    def calculate_diffusivity(self, data: pd.DataFrame) -> float:
        """Calculate diffusion coefficient [cm² s⁻¹]"""
        if self.params.transport.diffusivity is not None:
            return self.params.transport.diffusivity
            
        stab_time = find_stabilisation_time(data)
        time_lag, stats = find_time_lag(data, stab_time)
        
        D = self.params.base.thickness**2 / (6 * time_lag)
        
        self.results['time_lag'] = time_lag
        self.results['stabilisation_time'] = stab_time
        self.results['diffusivity'] = D
        
        return D

    def calculate_permeability(self, data: pd.DataFrame) -> float:
        """Calculate permeability [cm³(STP) cm⁻¹ s⁻¹ bar⁻¹]"""
        if self.params.transport.permeability is not None:
            return self.params.transport.permeability
            
        stab_time = self.results.get('stabilisation_time') or find_stabilisation_time(data)
        steady_state = data[data['time'] >= stab_time]
        
        slope = np.polyfit(steady_state['time'], steady_state['cumulative_flux'], 1)[0]
        P = slope * self.params.base.thickness / self.params.base.pressure
        
        self.results['permeability'] = P
        return P

    def calculate_solubility_coefficient(self) -> float:
        """Calculate solubility coefficient [cm³(STP) cm⁻³ bar⁻¹]"""
        if self.params.transport.solubility_coefficient is not None:
            return self.params.transport.solubility_coefficient
            
        if 'permeability' not in self.results or 'diffusivity' not in self.results:
            raise ValueError("Calculate diffusivity and permeability first")
            
        S = self.results['permeability'] / self.results['diffusivity']
        self.results['solubility_coefficient'] = S
        return S

    def calculate_equilibrium_concentration(self) -> float:
        """Calculate equilibrium concentration [cm³(STP) cm⁻³]"""
        if self.params.transport.equilibrium_concentration is not None:
            return self.params.transport.equilibrium_concentration
            
        C = self.results['solubility_coefficient'] * self.params.base.pressure
        self.results['equilibrium_concentration'] = C
        return C

    def solve_pde(self, D: float, C_eq: float, L: float, T: float, 
                  dt: float, dx: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Solve diffusion PDE with constant diffusivity
        
        Parameters
        ----------
        D : float
            Diffusion coefficient [cm² s⁻¹]
        C_eq : float
            Equilibrium concentration [cm³(STP) cm⁻³]
        L : float
            Membrane thickness [cm]
        T : float
            Total time [s]
        dt : float
            Time step [s]
        dx : float
            Spatial step [cm]
        """
        if dt > dx**2 / (2 * D):
            raise ValueError("Stability condition not met: dt <= dx²/(2D)")
        
        Nx = int(L / dx) + 1
        Nt = int(T / dt) + 1
        
        x = np.linspace(0, L, Nx)
        t = np.linspace(0, T, Nt)
        C = np.zeros(Nx)
        C_surface = np.zeros((Nt, Nx))
        flux_values = np.zeros(Nt)
        
        C[0] = C_eq
        
        for n in range(Nt):
            C_new = C.copy()
            
            for i in range(1, Nx-1):
                C_new[i] = C[i] + D * dt / dx**2 * (C[i+1] - 2*C[i] + C[i-1])
            
            C_new[0] = C_eq
            C_new[-1] = 0
            
            C = C_new.copy()
            C_surface[n, :] = C
            flux_values[n] = -D * (C[-1] - C[-2]) / dx
        
        df_C = pd.DataFrame(C_surface, columns=[f'x={x_i:.3f}' for x_i in x])
        df_C.index = t
        df_flux = pd.DataFrame({'time': t, 'flux': flux_values})
        
        return df_C, df_flux