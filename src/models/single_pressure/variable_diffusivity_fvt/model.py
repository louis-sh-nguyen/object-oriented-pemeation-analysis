from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ...base_model import PermeationModel
from ...base_parameters import BaseParameters
from ....utils.data_processing import preprocess_data
from .parameters import FVTModelParameters, FVTTransportParams
from ....utils.optimisation import OptimisationCallback

class FVTModel(PermeationModel):
    """Variable diffusivity model based on Free Volume Theory (FVT)."""
    
    def __init__(self, params: FVTModelParameters):
        super().__init__(params)
        self.area = np.pi * (params.transport.diameter/2)**2  # [cm²]
        self.results: Dict[str, Any] = {}
    
    @classmethod
    def from_parameters(cls,
                       pressure: float,
                       temperature: float,
                       thickness: float,
                       diameter: float,
                       flowrate: float,
                       D1_prime: Optional[float] = None,
                       DT_0: Optional[float] = None) -> 'FVTModel':
        """
        Create model instance from parameters
        
        Parameters
        ----------
        pressure : float
            Applied pressure [bar]
        temperature : float
            Temperature [°C]
        thickness : float
            Membrane thickness [cm]
        diameter : float
            Membrane diameter [cm]
        flowrate : float
            Flow rate [cm³(STP) min⁻¹]
        D1_prime : float, optional
            Normalized diffusivity at x=0 [adim]
        DT_0 : float, optional
            Temperature-dependent diffusivity [cm² s⁻¹]
        """
        base_params = BaseParameters(
            pressure=pressure,
            temperature=temperature
        )
        
        transport_params = FVTTransportParams(
            thickness=thickness,
            diameter=diameter,
            flowrate=flowrate,
            D1_prime=D1_prime,
            DT_0=DT_0
        )
        
        return cls(FVTModelParameters(base=base_params, transport=transport_params))
    
    def _solve_pde(self, D1_prime: float, DT_0: float, 
                   T: float, X: float, L: float, dt: float, dx: float, U_VprimeW: float = None, D2_prime: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Internal method to solve the Free Volume Theory (FVT) diffusion equation:
        ∂D'/∂t = (D0(T)/L²) * D' * ∂²D'/∂x²
        
        Parameters
        ----------
        D1_prime : float
            Normalized diffusivity at upstream boundary (x=0) [dimensionless]
            D'(x=0) = D1' = exp(-U/(V'W))
        DT_0 : float
            Temperature-dependent diffusion coefficient [cm² s⁻¹]
            D0(T) = D0 * exp(-Ed/RT)
        U_VprimeW : float
            Energy barrier parameter [dimensionless]
            U/(V'W) = ln(D1')
        T : float
            Total simulation time [s]
        X : float
            Normalized membrane thickness [dimensionless]
            x' = x/L where L is the membrane thickness
        L : float
            Membrane thickness [cm]
        dt : float
            Time step [s]
        dx : float
            Normalized spatial step [dimensionless]
        D2_prime : float, optional
            Normalized diffusivity at downstream boundary (x=L) [dimensionless]
            Default is 1.0 (no concentration dependence at x=L)
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            - Diffusivity profile D'(x,t) [dimensionless]
            - Normalized flux F'(t) = -∂D'/∂x at x=L [dimensionless]
        
        Notes
        -----
        The PDE is solved using an explicit finite difference method.
        Stability condition: dt ≤ dx²/(2*D0(T)/L²*D')
        """
        # Define spatial and temporal grids
        x = np.arange(0, X+dx, dx)
        t = np.arange(0, T+dt, dt)
        
        # Initialize D' profile
        D_prime = np.zeros((len(t), len(x)))  # (time, space)
        
        # Set initial and boundary conditions
        D_prime[0, :] = 1.0     # t=0
        D_prime[:, 0] = D1_prime     # x=0
        D_prime[:, -1] = D2_prime    # x=L
        
        # Solve PDE
        for n in range(1, len(t)):
            for i in range(1, len(x)-1):
                d2Dprime_dx2 = (D_prime[n-1, i+1] - 2*D_prime[n-1, i] + D_prime[n-1, i-1]) / dx**2
                D_prime[n, i] = D_prime[n-1, i] + dt * (DT_0 / L**2 * D_prime[n-1, i] * d2Dprime_dx2)
        
        # Calculate flux
        F_prime = -(D_prime[:, -1] - D_prime[:, -2]) / dx
        
        # Calculate normalised flux
        F_norm = F_prime / max(F_prime)
        
        # Calculate normalised time
        tau = DT_0 * t / L**2   # [adim]
        
        # Create DataFrames
        Dprime_df = pd.DataFrame(D_prime, columns=[f'x={x_i:.3f}' for x_i in x])
        flux_df = pd.DataFrame({'time': t, 
                                'flux': F_prime, 
                                'tau': tau,
                                'normalised_flux': F_norm,
                                })
        
        return Dprime_df, flux_df
    
    def solve_pde(self, simulation_params: Optional[Dict] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Solve PDE using model parameters
        
        Parameters
        ----------
        simulation_params : dict, optional
            Dictionary containing simulation parameters:
            - T: total time [s]
            - dt: time step [s]
            - dx: spatial step [normalized]
            - X: normalized position
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Diffusivity profile and flux data
        """
        if simulation_params is None:
            simulation_params = {
                'T': 100000,
                'dt': 1.0,
                'dx': 0.01,
                'X': 1.0
            }
        
        return self._solve_pde(
            D1_prime=self.params.transport.D1_prime,
            D2_prime=1.0,
            DT_0=self.params.transport.DT_0,
            T=simulation_params['T'],
            X=simulation_params['X'],
            L=self.params.transport.thickness,
            dt=simulation_params['dt'],
            dx=simulation_params['dx']
        )
    
    def fit_to_data(self, data: pd.DataFrame, 
                    track_progress: bool = False) -> Dict[str, float]:
        """
        Fit model parameters to experimental data
        
        Parameters
        ----------
        data : pd.DataFrame
            Experimental flux data with 'time' and 'flux' columns
        simulation_params : dict, optional
            Simulation parameters for PDE solving
        track_progress : bool, optional
            Whether to display optimization progress (default: False)
            
        Returns
        -------
        Dict[str, float]
            Fitted parameters (D1_prime, DT_0) and optimization results
        """
        # Check for required columns
        required_cols = ['tau', 'normalised_flux']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Data is missing required columns: {missing_cols}")

        # Setup optimization tracking if requested
        if track_progress:
            callback = OptimisationCallback(
                param_names=['D1_prime', 'DT_0'],
            )
        else:
            callback = None
            
        # Store last objective value for callback
        last_rmse = [float('inf')]
        
        def objective(params):
            D1_prime, DT_0 = params
            _, flux_df = self._solve_pde(
                D1_prime=D1_prime,
                DT_0=DT_0,
                L=self.params.transport.thickness,
                T=data['time'].max(),
                X=1.0,
                dt=data['time'].max() / 10000, # 10000 points
                dx=1.0 / 100,   # 100 points
            )
            
            # Interpolate model norm flux to data time points
            model_norm_flux = np.interp(data['tau'], flux_df['tau'], flux_df['normalised_flux'])
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((model_norm_flux - data['normalised_flux'])**2))
            last_rmse[0] = rmse
            
            return rmse
        
        def minimize_callback(xk):
            if callback is not None:
                callback(xk, last_rmse[0])
        
        # Initial guess from current parameters
        x0 = [self.params.transport.D1_prime, self.params.transport.DT_0]
        
        # Optimize with callback
        result = minimize(
            objective, 
            x0, 
            # method='Nelder-Mead',
            method='BFGS',
            callback=minimize_callback if callback is not None else None
        )
        
        # Update model parameters
        self.params.transport.D1_prime = result.x[0]
        self.params.transport.DT_0 = result.x[1]
        
        fit_results = {
            'D1_prime': result.x[0],
            'DT_0': result.x[1],
            'rmse': result.fun
        }
        
        # Add optimization history if tracked
        if track_progress and callback is not None:
            fit_results['optimization_history'] = callback.history
        
        return fit_results