from typing import Dict, Any, Tuple, Optional, Callable
import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.integrate import solve_ivp

from ...base_model import PermeationModel
from ...base_parameters import BaseParameters
from .parameters import TimelagModelParameters, TimelagTransportParams
from ....utils.time_analysis import find_stabilisation_time, find_time_lag, identify_start_time

class TimelagModel(PermeationModel):
    """Time-lag analysis model for constant diffusivity permeation"""
    
    def __init__(self, params: TimelagModelParameters):
        super().__init__(params)
        self.area = np.pi * (params.transport.diameter/2)**2  # [cm²]
        self.results: Dict[str, Any] = {}
    
    @classmethod
    def from_parameters(cls,
                       pressure: float,
                       temperature: float,
                       thickness: float,
                       diameter: float,
                       flowrate: Optional[float] = None,
                       diffusivity: Optional[float] = None,
                       equilibrium_concentration: Optional[float] = None) -> 'TimelagModel':
        """Create model instance from parameters"""
        base_params = BaseParameters(
            pressure=pressure,
            temperature=temperature
        )
        
        transport_params = TimelagTransportParams(
            thickness=thickness,
            diameter=diameter,
            flowrate=flowrate,
            diffusivity=diffusivity,
            equilibrium_concentration=equilibrium_concentration
        )
        
        return cls(TimelagModelParameters(base=base_params, transport=transport_params))

    def fit_to_data(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """Fit model to experimental data"""
        
        self.calculate_diffusivity(processed_data)
        self.calculate_permeability(processed_data)
        self.calculate_solubility_coefficient()
        self.calculate_equilibrium_concentration()
        
        return processed_data
    
    def calculate_diffusivity(self, data: pd.DataFrame) -> float:
        """Calculate diffusion coefficient [cm² s⁻¹]"""
        if self.params.transport.diffusivity is not None:
            return self.params.transport.diffusivity
        
        # Start of linear region of cumulative flux
        stab_start_time = identify_start_time(data, column='cumulative_flux')
        time_lag, stats = find_time_lag(data, stab_start_time)
        
        D = self.params.transport.thickness**2 / (6 * time_lag)
        
        self.results.update({
            'time_lag': time_lag,
            'stabilisation_time': stab_start_time,
            'diffusivity': D
        })
        
        return D

    def calculate_permeability(self, data: pd.DataFrame) -> float:
        """Calculate permeability [cm³(STP) cm⁻¹ s⁻¹ bar⁻¹]"""
        if self.params.transport.permeability is not None:
            return self.params.transport.permeability
            
        stab_time = self.results.get('stabilisation_time') or find_stabilisation_time(data)
        steady_state = data[data['time'] >= stab_time]
        
        # Calculate regression for steady state region
        slope, intercept, r_value, p_value, std_err = linregress(
            steady_state['time'], 
            steady_state['cumulative_flux']
        )
        
        P = slope * self.params.transport.thickness / self.params.base.pressure
        
        # Store regression results
        self.results.update({
            'permeability': P,
            'steady_state_slope': slope,
            'steady_state_intercept': intercept,
            'steady_state_r_squared': r_value**2,
            'steady_state_std_err': std_err
        })
        
        return P

    def get_steady_state_line(self, times: np.ndarray) -> np.ndarray:
        """Get steady state line for plotting"""
        if 'steady_state_slope' not in self.results or 'steady_state_intercept' not in self.results:
            raise ValueError("Calculate permeability first to get steady state line")
            
        return (self.results['steady_state_slope'] * times + 
                self.results['steady_state_intercept'])

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
            
        if 'permeability' not in self.results or 'diffusivity' not in self.results:
            raise ValueError("Calculate diffusivity and permeability first")
            
        C = self.results['solubility_coefficient'] * self.params.base.pressure
        self.results['equilibrium_concentration'] = C
        return C

    def _solve_pde(self, D: float, C_eq: float, L: float, T: float, 
                  N: int = 50, t_eval: Optional[np.ndarray] = None, 
                  method: str = 'BDF', rtol: float = 1e-4, atol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Helper method to solve diffusion PDE using solve_ivp
        
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
        N : int
            Number of spatial grid points
        t_eval : Optional[np.ndarray]
            Time points at which to store the solution. If None, defaults to 100 evenly spaced points.
        method : str
            Integration method to use in solve_ivp
        rtol : float
            Relative tolerance for the solver
        atol : float
            Absolute tolerance for the solver
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Time points, spatial points, concentration matrix, and flux values
        """
        # Spatial discretization
        x = np.linspace(0, L, N)
        dx = x[1] - x[0]
        
        # Time evaluation points
        if t_eval is None:
            t_eval = np.linspace(0, T, 100)
        
        # Initial condition (zero everywhere except at boundaries)
        C0 = np.zeros(N-2)  # Exclude boundary points
        
        # Define ODE system (discretized PDE)
        def diffusion_system(t, C):
            # Initialize with boundary conditions
            C_full = np.zeros(N)
            C_full[0] = C_eq  # Left boundary at x=0
            C_full[1:-1] = C  # Interior points
            C_full[-1] = 0    # Right boundary at x=L
            
            # Calculate spatial derivatives using central difference
            dC_dt = np.zeros(N-2)
            for i in range(N-2):
                dC_dt[i] = D * (C_full[i] - 2*C_full[i+1] + C_full[i+2]) / dx**2
            
            return dC_dt
        
        # Solve the system
        solution = solve_ivp(
            fun=diffusion_system,
            t_span=(0, T),
            y0=C0,
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            atol=atol
        )
        
        if not solution.success:
            raise RuntimeError(f"Integration failed: {solution.message}")
        
        # Extract results
        t = solution.t
        C_interior = solution.y
        
        # Create full concentration arrays with boundary conditions
        C_full = np.zeros((len(t), N))
        C_full[:, 0] = C_eq  # Left boundary
        C_full[:, 1:-1] = C_interior.T
        C_full[:, -1] = 0    # Right boundary
        
        # Calculate flux at x=L
        flux_values = np.zeros(len(t))
        for i in range(len(t)):
            # Use backward difference for flux at boundary
            flux_values[i] = -D * (C_full[i, -1] - C_full[i, -2]) / dx
        
        return t, x, C_full, flux_values
    
    def solve_pde(self, D: float, C_eq: float, L: float, T: float, 
                  dt: float, dx: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Solve diffusion PDE with constant diffusivity using solve_ivp
        
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
        # Calculate number of spatial points based on dx
        N = int(L / dx) + 1
        
        # Generate time evaluation points to match the original dt spacing
        Nt = int(T / dt) + 1
        t_eval = np.linspace(0, T, Nt)
        
        # Solve PDE using helper method
        t, x, C_full, flux_values = self._solve_pde(
            D=D, C_eq=C_eq, L=L, T=T, N=N, t_eval=t_eval
        )
        
        # Create DataFrames
        df_C = pd.DataFrame(C_full, columns=[f'x={x_i:.3f}' for x_i in x])
        df_C.index = t
        df_flux = pd.DataFrame({'time': t, 'flux': flux_values})
        
        return df_C, df_flux