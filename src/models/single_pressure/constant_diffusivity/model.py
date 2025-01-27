from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import linregress

from ...base_model import PermeationModel
from ...parameters import BaseParameters, ModelParameters
from ....utils.time_analysis import find_stabilisation_time, find_time_lag

class TimelagModel(PermeationModel):
    """
    Time-lag analysis model for constant diffusivity permeation.
    
    Methods:
    --------
    fit_to_data: Fit model to experimental data
    calculate_diffusivity: Calculate diffusion coefficient
    calculate_permeability: Calculate permeability
    calculate_solubility: Calculate solubility coefficient
    """
    
    def fit_to_data(self, data: pd.DataFrame) -> None:
        """Fit model to experimental data"""
        self.calculate_diffusivity(data)
        self.calculate_permeability(data)
        self.calculate_solubility_coefficient()
        self.calculate_solubility()
        
        # Update model parameters with fitted values
        self.params.diffusivity = self.results['diffusivity']
        self.params.permeability = self.results['permeability']
        self.params.solubility_coefficient = self.results['solubility_coefficient']
        self.params.solubility = self.results['solubility']
    
    def calculate_diffusivity(self, data: pd.DataFrame) -> float:
        """Calculate diffusion coefficient [cm² s⁻¹]"""
        if self.params.diffusivity is not None:
            return self.params.diffusivity
            
        stab_time = find_stabilisation_time(data)
        time_lag, _ = find_time_lag(data, stab_time)
        
        D = self.params.base.thickness**2 / (6 * time_lag)
        
        self.results['time_lag'] = time_lag
        self.results['stabilisation_time'] = stab_time
        self.results['diffusivity'] = D
        
        return D
    
    def calculate_permeability(self, data: pd.DataFrame) -> float:
        """Calculate permeability [cm³(STP) cm⁻¹ s⁻¹ bar⁻¹]"""
        if self.params.permeability is not None:
            return self.params.permeability
            
        stab_time = self.results.get('stabilisation_time') or find_stabilisation_time(data)
        steady_state = data[data['time'] >= stab_time]
        
        slope = np.polyfit(steady_state['time'], steady_state['cumulative flux'], 1)[0]    # update to use cumulative flux
        P = slope * self.params.base.thickness / self.params.base.pressure
        
        self.results['permeability'] = P
        return P
    
    def calculate_solubility_coefficient(self) -> float:
        """Calculate solubility coefficient [cm³(STP) cm⁻³ bar⁻¹]"""
        if self.params.solubility is not None:
            return self.params.solubility
            
        if 'permeability' not in self.results or 'diffusivity' not in self.results:
            raise ValueError("Calculate diffusivity and permeability first")
            
        S = self.results['permeability'] / self.results['diffusivity']
        self.results['solubility_coefficient'] = S
        return S
    
    def calculate_solubility(self) -> float:
        """Calculate solubility coefficient [cm³(STP) cm⁻³ bar⁻¹]"""
        if self.params.solubility is not None:
            return self.params.solubility
            
        if 'permeability' not in self.results or 'diffusivity' not in self.results:
            raise ValueError("Calculate diffusivity and permeability first")
            
        C = self.results['solubility_coefficient'] * self.params.base.pressure
        self.results['solubility'] = C
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
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Concentration profile and flux results
        """
        # Validate parameters
        if dt > dx**2 / (2 * D):
            raise ValueError("Stability condition not met: dt <= dx²/(2D)")
        
        # Calculate grid points
        Nx = int(L / dx) + 1
        Nt = int(T / dt) + 1
        
        # Initialize arrays
        x = np.linspace(0, L, Nx)
        t = np.linspace(0, T, Nt)
        C = np.zeros(Nx)
        C_surface = np.zeros((Nt, Nx))
        flux_values = np.zeros(Nt)
        
        # Initial condition
        C[0] = C_eq  # Boundary condition at x=0
        
        # Time stepping
        for n in range(Nt):
            C_new = C.copy()
            
            # Space stepping
            for i in range(1, Nx-1):
                C_new[i] = C[i] + D * dt / dx**2 * (C[i+1] - 2*C[i] + C[i-1])
            
            # Boundary conditions
            C_new[0] = C_eq
            C_new[-1] = 0
            
            # Store results
            C = C_new.copy()
            C_surface[n, :] = C
            
            # Calculate flux at x=L
            flux_values[n] = -D * (C[-1] - C[-2]) / dx
        
        # Create DataFrames
        df_C = pd.DataFrame(C_surface, columns=[f'x={x_i:.3f}' for x_i in x])
        df_C.index = t
        df_flux = pd.DataFrame({'time': t, 'flux': flux_values})
        
        return df_C, df_flux