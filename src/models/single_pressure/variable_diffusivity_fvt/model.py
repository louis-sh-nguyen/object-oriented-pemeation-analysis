from typing import Tuple, Optional
import numpy as np
import pandas as pd

from ...base_model import PermeationModel
from ...parameters import BaseParameters, ModelParameters, TransportParams
from ....utils.time_analysis import find_stabilisation_time, find_time_lag
from ....utils.data_processing import preprocess_data

class FVTModel(PermeationModel):
    """
    Variable diffusivity model based on Free Volume Theory (FVT).
    
    Methods:
    --------
    fit_to_data: Fit model to experimental data
    calculate_diffusivity: Calculate diffusion coefficient
    calculate_permeability: Calculate permeability
    calculate_solubility: Calculate solubility coefficient
    """
    def __init__(self, params: ModelParameters):
        """
        Initialise FVTModel.

        Parameters
        ----------

        params : ModelParameters
            Model parameters containing base parameters and optional diffusivity and equilibrium concentration
        """
        super().__init__(params)
        self.results = {}
    
    @classmethod
    def from_parameters(cls,
                    thickness: float,
                    diameter: float,
                    flowrate: float,
                    pressure: float,
                    temperature: float = 25.0,
                    D1_prime: Optional[float] = None,
                    D2_prime: Optional[float] = None,
                    D0_T: Optional[float] = None,) -> 'FVTModel':
        """
        Create model instance from parameters
        
        Parameters
        ----------
        thickness : float
            Membrane thickness [cm]
        diameter : float
            Membrane diameter [cm]
        flowrate : float
            Flow rate [cm³(STP) s⁻¹]
        pressure : float
            Pressure [bar]
        temperature : float
            Temperature [°C]
        D1_prime : float, optional
            Base diffusivity [cm² s⁻¹]
        D2_prime : float, optional
            Concentration-dependent diffusivity [cm² s⁻¹ cm³(STP)/cm³]
        """
        base_params = BaseParameters(
            thickness=thickness,
            diameter=diameter,
            flowrate=flowrate,
            pressure=pressure,
            temperature=temperature
        )
        
        transport_params = TransportParams(
            D1_prime=D1_prime,
            D2_prime=D2_prime,
            D0_T = D0_T
        )
        
        return cls(ModelParameters(base=base_params, transport=transport_params))
    
    def solve_pde(self, D1_prime: float, D2_prime: float, D0_T: float, L: float, T: float, dt: float, dx: float) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Solve concentration profile using finite volume method.
        
        Parameters
        ----------
        D1_prime : float
            Base diffusivity [cm² s⁻¹]
        D2_prime : float
            Concentration-dependent diffusivity [cm² s⁻¹ cm³(STP)/cm³]
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
        np.ndarray
            Concentration profile
        pd.DataFrame
            Flux data
        """
        # Define spatial and temporal grids
        x = np.arange(0, L+dx, dx)
        t = np.arange(0, T+dt, dt)
        
        # Initialize D' profile
        D_prime = np.zeros((len(t), len(x)))  # (time, space)
        
        # Set initial condition
        D_prime[0, :] = 0.0     # t=0
        
        # Set boundary conditions
        D_prime[:, 0] = D1_prime     # x=0
        D_prime[:, -1] = D2_prime    # x=L
        
        # Solve PDE
        for n in range(1, len(t)):  # skip t=0
            for i in range(1, len(x)-1):    # skip x=0 and x=L
                d2Dprime_dx2 = (D_prime[n-1, i+1] - 2*D_prime[n-1, i] + D_prime[n-1, i-1]) / dx**2
                D_prime[n, i] = D_prime[n-1, i] + dt * (D0_T / L**2 * D_prime[n, i] * d2Dprime_dx2)
        
        # Boundary conditions
        D_prime[:, 0] = 1.0
        D_prime[:, -1] = 0.0
        
        # Calculate flux
        # F_prime = np.zeros(len(t))
        F_prime = -(D_prime[:, -1] - D_prime[:, -2]) / dx
        
        # Dataframe
        Dprime_df = pd.DataFrame(D_prime, columns=[f'x={x_i:.3f}' for x_i in x])
        flux_df = pd.DataFrame({'time': t, 'flux': F_prime})
        
        return Dprime_df, flux_df