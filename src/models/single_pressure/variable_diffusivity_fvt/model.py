from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import minimize
from tqdm import tqdm
import numba as nb

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
    
    # def _solve_pde(self, D1_prime: float, DT_0: float, T: float, X: float, L: float, dt: float, dx: float, U_VprimeW: float = None, D2_prime: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #     """Solve PDE with stability monitoring
        
    #     ∂D'/∂t = (DT_0/L²) * D' * (∂²D'/∂x²)
        
    #     Finite Difference Scheme
    #     - Forward difference in time
    #     - Central difference in space
        
    #     Discretization
    #     ∂D'/∂t ≈ (D'[n+1,i] - D'[n,i])/dt
    #     ∂²D'/∂x² ≈ (D'[n,i+1] - 2D'[n,i] + D'[n,i-1])/dx²

    #     Stability Criterion
    #     dt ≤ dx²/(2*DT_0*D1')
                
    #     Matrix form:
    #     D'[n+1] = D'[n] + r * D'[n] * (AD'[n])
    #     Where:
    #     - r = dt*DT_0/L²
    #     - A: Tridiagonal matrix for ∂²/∂x²
        
    #     Known undersirable behaviours:
    #     - dx <= 0.01 -> unstable.
    #     - d2D = laplacian.dot(D_prime[n-1, 1:-1]) -> causes internal points to drop to 0 at t > 0

        
    #     """
    #     # Grid setup
    #     Nx = int(X/dx) + 1
    #     Nt = int(T/dt) + 1
        
    #     # Initialize arrays
    #     x = np.linspace(0, X, Nx)
    #     t = np.linspace(0, T, Nt)
    #     D_prime = np.zeros((Nt, Nx))
        
    #     # Initialise
    #     D_prime[0, :] = 1.0  # Example initial condition, modify as necessary
        
    #     # Boundary conditions
    #     D_prime[:, 0] = D1_prime     # x=0
    #     D_prime[:, -1] = D2_prime    # x=L
        
    #     # Pre-compute constants
    #     r = dt * DT_0 / (L**2)
        
    #     # Esitmate second derivative operator (sparse matrix) ưith laplacian
    #     Nx_interior = Nx - 2  # Number of interior points
    #     diagonals = [np.ones(Nx_interior-1), -2*np.ones(Nx_interior), np.ones(Nx_interior-1)]
    #     laplacian = sparse.diags(diagonals, offsets=[-1, 0, 1], shape=(Nx_interior, Nx_interior), format='csr') / dx**2
    
    #     # Time stepping with progress bar
    #     # Progress bar
    #     pbar = tqdm(
    #         total=Nt-1,
    #         desc=f"Solving PDE (D1'={D1_prime:.2f}, DT_0={DT_0:.2e})",
    #         unit='steps',
    #         ncols=100
    #     )
        
    #     # Time stepping
    #     for n in range(1, Nt):
    #         # Internal-point stepping
    #         for i in range(1, Nx-1):
    #             d2D = (D_prime[n-1, i+1] - 2 * D_prime[n-1, i] + D_prime[n-1, i-1]) / dx**2     # stable
    #             D_prime[n, i] = D_prime[n-1, i] + r * D_prime[n-1, i] * d2D
            
    #         pbar.update(1)
        
    #     pbar.close()
        
    #     # Calculate flux
    #     F_norm = (-(D_prime[:, -1] - D_prime[:, -2]) / dx) / (-(D2_prime - D1_prime) / X)
        
    #     # Create output DataFrames
    #     Dprime_df = pd.DataFrame(D_prime, columns=[f'x={x_i:.3f}' for x_i in x])
    #     Dprime_df.index = t
        
    #     flux_df = pd.DataFrame({
    #         'time': t,
    #         'normalised_flux': F_norm,
    #         'tau': DT_0 * t / L**2
    #     })
        
    #     return Dprime_df, flux_df
    
    # def _solve_pde(self, D1_prime: float, DT_0: float, T: float, X: float, L: float, dt: float, dx: float, U_VprimeW: float = None, D2_prime: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #     """Solve PDE with implicit Euler scheme (backward Euler).

    #     """
    #     # Check D1_prime and D2_prime
    #     if D1_prime <= 0 or D2_prime <= 0:
    #         raise ValueError(f'D1_prime ({D1_prime}) and D2_prime ({D2_prime}) must be positive')
    #     if D1_prime <= D2_prime:
    #         raise ValueError(f'D1_prime ({D1_prime}) must be greater than D2_prime ({D2_prime})')
        
    #     # Grid setup
    #     Nx = int(X/dx) + 1
    #     Nt = int(T/dt) + 1
        
    #     # Initialize arrays
    #     x = np.linspace(0, X, Nx)
    #     t = np.linspace(0, T, Nt)
        
    #     # Initial condition
    #     D_prime = np.ones((Nt, Nx))  # Initial condition
        
    #     # Boundary conditions
    #     D_prime[:, 0] = D1_prime     # x=0
    #     D_prime[:, -1] = D2_prime    # x=L
        
    #     # Pre-compute constants
    #     r = (DT_0 / L**2) * (dt / dx**2)
        
    #     # Construct tridiagonal matrix A for interior points (size Nx-2)
    #     diagonals = [
    #         -r * np.ones(Nx - 3),  # Lower diagonal
    #         (1 + 2 * r) * np.ones(Nx - 2),  # Main diagonal
    #         -r * np.ones(Nx - 3)   # Upper diagonal
    #     ]
    #     A = sp.diags(diagonals, offsets=[-1, 0, 1], format="csr")
    
    #     # Store flux values
    #     F_norm = np.zeros(Nt)
        
    #     # Time stepping with progress bar
    #     # Progress bar
    #     pbar = tqdm(
    #         total=Nt-1,
    #         desc=f"Solving PDE (D1'={D1_prime:.2f}, DT_0={DT_0:.2e})",
    #         unit='steps',
    #         ncols=100
    #     )
        
    #     # Time stepping
    #     for n in range(1, Nt):
    #         # Right-hand side (RHS)
    #         rhs = D_prime[n-1, 1:-1].copy()  # Only interior points
            
    #         # Apply boundary conditions explicitly to RHS
    #         rhs[0] += r * D_prime[n, 0]    # Left boundary
    #         rhs[-1] += r * D_prime[n, -1]  # Right boundary

    #         # Solve linear system for interior points
    #         D_prime[n, 1:-1] = spla.spsolve(A, rhs)

    #         # Calculate flux at right boundary
    #         F_norm[n] = (-(D_prime[n, -1] - D_prime[n, -2]) / dx) / (-(D_prime[n, -1] - D_prime[n, 0]) / X)
            
    #         pbar.update(1)
        
    #     pbar.close()
        
    #     # Create output DataFrames
    #     Dprime_df = pd.DataFrame(D_prime, columns=[f"x={x_i:.3f}" for x_i in x])
    #     Dprime_df.index = t
        
    #     flux_df = pd.DataFrame({
    #         'time': t,
    #         'normalised_flux': F_norm,
    #         'tau': DT_0 * t / L**2
    #     })
        
    #     return Dprime_df, flux_df

    # def _solve_pde(self, D1_prime: float, DT_0: float, T: float, X: float, L: float, dt: float, dx: float, U_VprimeW: float = None, D2_prime: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #     """Solve PDE with Crank-Nicholson (implicit).

    #     """
    #     # Check D1_prime and D2_prime
    #     if D1_prime <= 0 or D2_prime <= 0:
    #         raise ValueError(f'D1_prime ({D1_prime}) and D2_prime ({D2_prime}) must be positive')
    #     if D1_prime <= D2_prime:
    #         raise ValueError(f'D1_prime ({D1_prime}) must be greater than D2_prime ({D2_prime})')
        
    #     # Grid setup
    #     Nx = int(X/dx) + 1
    #     Nt = int(T/dt) + 1
        
    #     # Initialize arrays
    #     x = np.linspace(0, X, Nx)
    #     t = np.linspace(0, T, Nt)
        
    #     # Initial condition
    #     D_prime = np.ones((Nt, Nx))  # Initial condition
        
    #     # Boundary conditions
    #     D_prime[:, 0] = D1_prime     # x=0
    #     D_prime[:, -1] = D2_prime    # x=L
        
    #     # Pre-compute constants
    #     r = (DT_0 / L**2) * (dt / dx**2)
        
    #     # Construct tridiagonal matrix A for interior points (size Nx-2)
    #     diagonals = [
    #         -r * np.ones(Nx - 3),  # Lower diagonal
    #         (1 + 2 * r) * np.ones(Nx - 2),  # Main diagonal
    #         -r * np.ones(Nx - 3)   # Upper diagonal
    #     ]
    #     A = sp.diags(diagonals, offsets=[-1, 0, 1], format="csr")
    
    #     # Store flux values
    #     F_norm = np.zeros(Nt)
        
    #     # Time stepping with progress bar
    #     # Progress bar
    #     pbar = tqdm(
    #         total=Nt-1,
    #         desc=f"Solving PDE (D1'={D1_prime:.2f}, DT_0={DT_0:.2e})",
    #         unit='steps',
    #         ncols=100
    #     )
    #     try:
    #         for n in range(1, Nt):
    #             # Update interior points using implicit Euler method
    #             b = D_prime[n-1, 1:-1].copy()
                
    #             # Apply boundary conditions explicitly to RHS
    #             b[0] += r * D_prime[n, 0]    # Left boundary
    #             b[-1] += r * D_prime[n, -1]  # Right boundary
                
    #             D_new = spsolve(A, b)
                
    #             # Check for instabilities
    #             if not np.all(np.isfinite(D_new)):
    #                 raise ValueError(f"Solution became unstable at t={t[n]:.2e}")
                
    #             D_prime[n, 1:-1] = D_new
    #             pbar.update(1)
                
    #             if n % 100 == 0:
    #                 pbar.set_postfix({
    #                     'tau': f"{DT_0 * t[n] / L**2:.2e}",
    #                     'max_D': f"{np.max(D_new):.2e}"
    #                 })
        
    #     except Exception as e:
    #         pbar.close()
    #         raise e
    #     pbar.close()
        
    #     # Create output DataFrames
    #     Dprime_df = pd.DataFrame(D_prime, columns=[f"x={x_i:.3f}" for x_i in x])
    #     Dprime_df.index = t
        
    #     flux_df = pd.DataFrame({
    #         'time': t,
    #         'normalised_flux': F_norm,
    #         'tau': DT_0 * t / L**2
    #     })
        
    #     return Dprime_df, flux_df

    @staticmethod
    @nb.njit
    def _newton_update_jit(D_old, dt, dx, K, max_iter, D1_prime, D2_prime, relax, rel_tol):
        """
        JIT-compiled helper to perform Newton iterations for one time step.
        Returns updated solution D_new and a convergence flag (1 if converged, else 0).
        """
        D_new = D_old.copy()
        Nx = D_old.shape[0]
        converged = 0
        dx2 = dx * dx
        for it in range(max_iter):
            R = np.empty(Nx - 2, dtype=np.float64)
            for i in range(1, Nx - 1):
                lapl = (D_new[i+1] - 2.0 * D_new[i] + D_new[i-1]) / dx2
                R[i-1] = (D_new[i] - D_old[i]) / dt - K * D_new[i] * lapl

            sumR = 0.0
            sumD = 0.0
            for i in range(Nx - 2):
                sumR += R[i] * R[i]
            for i in range(1, Nx - 1):
                sumD += D_new[i] * D_new[i]
            norm_R = np.sqrt(sumR)
            norm_D = np.sqrt(sumD)
            if norm_D == 0.0:
                norm_D = 1e-12

            if norm_R / norm_D < rel_tol:
                converged = 1
                break

            for i in range(1, Nx - 1):
                lapl = (D_new[i+1] - 2.0 * D_new[i] + D_new[i-1]) / dx2
                J_diag = (1.0 / dt) - K * (lapl + (-2.0 * D_new[i]) / dx2)
                if J_diag == 0.0:
                    J_diag = 1e-8
                D_new[i] = D_new[i] - relax * (((D_new[i] - D_old[i]) / dt) - K * D_new[i] * lapl) / J_diag

            D_new[0] = D1_prime
            D_new[Nx - 1] = D2_prime
        return D_new, converged

    def _solve_pde(self, D1_prime, DT_0, T, X, L, dx, 
                  D2_prime=1.0, rel_tol=1e-8, max_iter=100, relax=0.8,
                  dt_init=0.0005, dt_target=10, dt_min=1e-6, dt_ramp_factor=1.1):
        """
        Adaptive implicit PDE solver using Newton's method with adaptive dt.
        Instead of stopping the simulation when dt becomes too small,
        the solver switches to fallback mode: dt is fixed to dt_min and damping is increased.
        
        Returns:
        Dprime_df, flux_df (solution history and flux history as DataFrames)
        """
        Nx = int(X / dx) + 1
        x = np.linspace(0, X, Nx)
        K = DT_0 / (L * L)
        D0 = np.ones(Nx, dtype=np.float64)
        D0[0] = D1_prime
        D0[-1] = D2_prime

        t_history = [0.0]
        D_history = [D0.copy()]
        current_t = 0.0
        dt = dt_init

        pbar = tqdm(total=T, desc=f"Adaptive PDE Solve (D1'={D1_prime}, DTO={DT_0})", ncols=100)
        while current_t < T:
            accepted = False
            trial_dt = dt
            current_relax = relax  # start with the original relaxation
            while not accepted:
                D_new, converged = FVTModel._newton_update_jit(D_history[-1], trial_dt, dx, K,
                                                    max_iter, D1_prime, D2_prime, current_relax, rel_tol)
                if converged == 1:
                    accepted = True
                else:
                    trial_dt *= 0.5
                    # print("Reducing dt to", trial_dt)
                    # Instead of stopping, if trial_dt drops below dt_min, switch to fallback.
                    if trial_dt < dt_min:
                        # print("dt has reached the minimum threshold. Switching to dt_min and increasing damping.")
                        trial_dt = dt_min
                        current_relax = current_relax * 0.5  # increase damping
            if trial_dt < dt_target:
                new_dt = min(trial_dt * dt_ramp_factor, dt_target)
                # if new_dt > trial_dt:
                #     print("Increasing dt from", trial_dt, "to", new_dt)
                dt = new_dt
            else:
                dt = trial_dt
            current_t += dt
            t_history.append(current_t)
            D_history.append(D_new.copy())
            pbar.update(dt) # increments the progress by an amount dt
        pbar.close()

        D_arr = np.array(D_history)
        t_arr = np.array(t_history)
        F_norm = (-(D_arr[:, -1] - D_arr[:, -2]) / dx) / (-(D2_prime - D1_prime) / X)
        
        Dprime_df = pd.DataFrame(D_arr, columns=[f"x={xi:.3f}" for xi in x], index=t_arr)
        flux_df = pd.DataFrame({
            "time": t_arr,
            "normalised_flux": F_norm,
            "tau": DT_0 * t_arr / (L * L)
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
        # Parameter validation
        if simulation_params is None:
            simulation_params = {
                'T': 10e3,
                'dx': 0.01,
                'X': 1.0
            }

        # Call _solve_pde
        return self._solve_pde(
            L=self.params.transport.thickness,
            D1_prime=self.params.transport.D1_prime,
            D2_prime=1.0,
            DT_0=self.params.transport.DT_0,
            T=simulation_params['T'],
            X=simulation_params['X'],
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
                L=self.params.transport.thickness,
                D1_prime=D1_prime,
                DT_0=DT_0,
                T=data['time'].max(),
                X=1.0,
                dt=1,
                dx=0.02,   # 100 points
            )
            
            # Interpolate model norm flux to data time points
            model_norm_flux = np.interp(data['tau'], flux_df['tau'], flux_df['normalised_flux'])
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((data['normalised_flux'] - model_norm_flux)**2))
            last_rmse[0] = rmse
            
            return rmse
        
        def minimize_callback(xk):
            if callback is not None:
                callback(xk, last_rmse[0])
        
        # Initial guess from current parameters
        x0 = [self.params.transport.D1_prime, self.params.transport.DT_0]
        bounds = [(0.1, 100), (1e-10, 1e-4)] # [(D1_min, D1_max), (DT0_min, DT0_max)]
        
        # Optimize with callback
        result = minimize(
            objective, 
            x0, 
            # method='Nelder-Mead',
            method='L-BFGS-B',
            bounds=bounds,
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