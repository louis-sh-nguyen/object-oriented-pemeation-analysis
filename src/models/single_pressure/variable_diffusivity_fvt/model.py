from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from tqdm import tqdm
import numba as nb

from ...base_model import PermeationModel
from ...base_parameters import BaseParameters
from .parameters import FVTModelParameters, FVTTransportParams
from ....utils.optimisation import OptimisationCallback
import time

@nb.njit
def _solve_tridiagonal(a, b, c, d):
    """Optimized Thomas algorithm implementation."""
    n = d.shape[0]
    # Use single arrays to minimize memory allocation
    c_prime = c.copy()
    d_prime = d.copy()
    
    # Forward sweep
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, n):
        m = 1.0 / (b[i] - a[i-1] * c_prime[i-1])
        c_prime[i] = c[i] * m if i < n-1 else 0.0
        d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) * m
    
    # Back substitution (in-place in d_prime)
    for i in range(n-2, -1, -1):
        d_prime[i] = d_prime[i] - c_prime[i] * d_prime[i+1]
    
    return d_prime

@nb.njit
def _newton_update_jit(D_old, dt, dx, K, max_iter, D1_prime, D2_prime, relax, rel_tol, use_full_jacobian):
    """Optimized Newton iteration implementation with option for diagonal/full Jacobian."""
    D_new = D_old.copy()
    Nx = D_old.shape[0]
    dx2 = dx * dx
    converged = 0
    
    # Pre-allocate arrays
    R = np.empty(Nx - 2, dtype=np.float64)
    J_diag = np.empty(Nx - 2, dtype=np.float64)
    if use_full_jacobian:
        J_lower = np.empty(Nx - 3, dtype=np.float64)
        J_upper = np.empty(Nx - 3, dtype=np.float64)
    
    for it in range(max_iter):
        # Combined residual and Jacobian calculation
        sumR = 0.0
        sumD = 0.0
        
        # Calculate residual and Jacobian elements in a single loop
        for i in range(1, Nx - 1):
            # Compute Laplacian directly
            lapl = (D_new[i+1] - 2.0 * D_new[i] + D_new[i-1]) / dx2
            
            # Calculate residual
            R[i-1] = (D_new[i] - D_old[i]) / dt - K * D_new[i] * lapl
            sumR += R[i-1] * R[i-1]
            sumD += D_new[i] * D_new[i]
            
            # Calculate Jacobian elements
            J_diag[i-1] = (1.0 / dt) - K * (lapl + (-2.0 * D_new[i]) / dx2)
            
            # Off-diagonal elements
            if use_full_jacobian:
                if i > 1:
                    J_lower[i-2] = K * D_new[i] / dx2
                if i < Nx - 2:
                    J_upper[i-1] = K * D_new[i] / dx2
        
        # Check convergence with efficient norm calculation
        norm_R = np.sqrt(sumR)
        norm_D = np.sqrt(sumD) if sumD > 0 else 1e-12
        
        if norm_R / norm_D < rel_tol:
            converged = 1
            break
        
        # Solve the tridiagonal system efficiently
        if use_full_jacobian:
            update = _solve_tridiagonal(J_lower, J_diag, J_upper, -R)
        else:
            update = -R / J_diag
        
        # Update solution
        for i in range(1, Nx - 1):
            D_new[i] += relax * (update[i-1] if use_full_jacobian else update[i-1])
        
        # Apply boundary conditions
        D_new[0] = D1_prime
        D_new[-1] = D2_prime
        
    return D_new, converged

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

    def _solve_pde(self, D1_prime, DT_0, T, X, L, dx, 
                  D2_prime=1.0, rel_tol=1e-8, atol=None,
                  dt_init=0.0005, track_solving_progress=True,
                  D0=None):  # Removed unused parameters
        """
        Solve the nonlinear PDE using scipy's solve_ivp method (Method of Lines).
        
        Parameters
        ----------
        D1_prime : float
            Normalized diffusivity at x=0 (boundary condition)
        DT_0 : float
            Temperature-dependent diffusivity coefficient [cm² s⁻¹]
        T : float
            Total simulation time [s]
        X : float
            Normalized spatial domain length (usually 1.0)
        L : float
            Membrane thickness [cm]
        dx : float
            Spatial step size for discretization
        D2_prime : float, optional
            Normalized diffusivity at x=X (boundary condition), default=1.0
        rel_tol : float, optional
            Relative tolerance for solver
        atol : float, optional
            Absolute tolerance for solver, defaults to rel_tol * 0.1 if None
        dt_init : float, optional
            Initial time step hint for solver
        track_solving_progress : bool, optional
            Whether to display progress information
        D0 : ndarray, optional
            Initial condition for D profile. If None, defaults to flat profile.
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Diffusivity profiles and flux data
        """
        # Discretize space
        Nx = int(X / dx) + 1
        x = np.linspace(0, X, Nx)
        K = DT_0 / (L * L)
        
        # Define the ODE system derived from Method of Lines
        def diffusion_ode(t, D):
            """Convert PDE to system of ODEs using Method of Lines"""
            dDdt = np.zeros_like(D)
            dx2 = dx * dx
            
            # Apply interior points dynamics
            for i in range(1, len(D)-1):
                lapl = (D[i+1] - 2.0*D[i] + D[i-1]) / dx2
                dDdt[i] = K * D[i] * lapl
            
            # Boundary conditions
            dDdt[0] = 0  # Fixed D1_prime (Dirichlet)
            dDdt[-1] = 0  # Fixed D2_prime (Dirichlet)
            
            return dDdt
        
        # Initial condition
        if D0 is None:
            D0 = np.ones(Nx, dtype=np.float64)
            D0[0] = D1_prime     # Left boundary
            D0[-1] = D2_prime    # Right boundary
        else:
            # Ensure boundary conditions are properly set in provided D0
            D0[0] = D1_prime
            D0[-1] = D2_prime
        
        # Configuration for the solver
        if atol is None:
            atol = rel_tol * 0.1  # Set absolute tolerance based on relative tolerance
        
        # Track solution timing if progress tracking is enabled
        if track_solving_progress:
            start_time = time.time()
            print(f"Starting solve_ivp integration (D1'={D1_prime:.2f}, DT0={DT_0:.2e})...")
            
        # Solve using solve_ivp with BDF method (good for stiff problems)
        solution = solve_ivp(
            diffusion_ode, 
            t_span=[0, T],
            y0=D0,
            method='BDF',  # Backward Differentiation Formula (for stiff problems)
            rtol=rel_tol,
            atol=atol,
            first_step=dt_init,  # Initial step hint
            max_step=T/10,  # Use reasonable max step based on total time
            dense_output=True  # Allow interpolation for output
        )
        
        if track_solving_progress:
            end_time = time.time()
            solve_time = end_time - start_time
            print(f"Integration complete. Time taken: {solve_time:.4f} seconds")
            print(f"Number of function evaluations: {solution.nfev}")
            print(f"Number of Jacobian evaluations: {solution.njev}")
            print(f"Solver status: {solution.status} (0=success)")
        
        # Extract solution information - get more points for smoother output
        num_output_points = min(1000, max(100, int(T/10)))  # Adaptive number of output points
        t_eval = np.linspace(0, T, num_output_points)
        D_history = []
        
        for t in t_eval:
            D_t = solution.sol(t)
            D_history.append(D_t)
        
        D_arr = np.array(D_history)
        
        # Calculate normalized flux
        F_norm = (-(D_arr[:, -1] - D_arr[:, -2]) / dx) / (-(D2_prime - D1_prime) / X)
        
        # Create DataFrames for output
        Dprime_df = pd.DataFrame(D_arr, columns=[f"x={xi:.3f}" for xi in x], index=t_eval)
        flux_df = pd.DataFrame({
            "time": t_eval,
            "normalised_flux": F_norm,
            "tau": DT_0 * t_eval / (L * L)
        })
        
        return Dprime_df, flux_df

    def solve_pde(self, simulation_params: Optional[Dict] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Solve PDE using model parameters.
        
        Parameters
        ----------
        simulation_params : dict, optional
            Dictionary containing simulation parameters:
            - T: total time [s]
            - dt: initial time step [s]
            - dx: spatial step [normalized]
            - X: normalized position
            - rel_tol: relative tolerance for convergence (default 1e-8)
            - atol: absolute tolerance for solver (default rel_tol*0.1)
            - fitting_mode: set to True when called from fitting functions (default False)
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Diffusivity profile and flux data
        """
        # Parameter validation
        if simulation_params is None:
            simulation_params = {}
            
        # Extract parameters with defaults
        rel_tol = simulation_params.get('rel_tol', 1e-8)
        atol = simulation_params.get('atol', None)
        fitting_mode = simulation_params.get('fitting_mode', False)
        
        # Solver with solve_ivp
        D_profile, flux = self._solve_pde(
            D1_prime=self.params.transport.D1_prime,
            DT_0=self.params.transport.DT_0,
            T=simulation_params.get('T', 10e3),
            X=simulation_params.get('X', 1.0),
            L=self.params.transport.thickness,
            dx=simulation_params.get('dx', 0.01),
            dt_init=simulation_params.get('dt', 0.001),
            rel_tol=rel_tol,
            atol=atol,
            track_solving_progress=not fitting_mode
        )
        
        self.results['D_profile'] = D_profile
        self.results['flux'] = flux
        return D_profile, flux

    def _process_fitting_settings(self, fitting_settings: Optional[dict]) -> tuple:
        """
        Process the fitting_settings argument and return a tuple
        (initial_guess, bounds, n_starts) to be passed to the helper functions.
        
        For mode "d1":
        - initial_guess: float (default: self.params.transport.D1_prime or 5.0)
        - bounds: tuple of two numbers (default: (1.01, 100))
        For mode "both":
        - initial_guess: tuple (default: (self.params.transport.D1_prime, self.params.transport.DT_0) or (5.0, 1e-6))
        - bounds: tuple of tuples (default: ((1.01, 100), (1e-8, 1e-5)))
        n_starts is an integer with default value 1.
        """
        n_starts = fitting_settings.get("n_starts", 1) if fitting_settings else 1
        exploitation_weight = fitting_settings.get("exploitation_weight", 0.7) if fitting_settings else 0.7
        
        mode = fitting_settings.get("mode", "D1") if fitting_settings else "D1"
        
        if mode == "both":
            default_init = (self.params.transport.D1_prime, self.params.transport.DT_0) if hasattr(self, "params") else (5.0, 1e-6)
            initial_guess = fitting_settings.get("initial_guess", default_init) if fitting_settings else default_init
            default_bounds = ((1.01, 100), (1e-8, 1e-5))
            bounds = fitting_settings.get("bounds", default_bounds) if fitting_settings else default_bounds
        elif mode == 'D1':
            default_init = self.params.transport.D1_prime if hasattr(self, "params") else 5.0
            initial_guess = fitting_settings.get("initial_guess", default_init) if fitting_settings else default_init
            default_bounds = (1.01, 100)
            bounds = fitting_settings.get("bounds", default_bounds) if fitting_settings else default_bounds
        
        return initial_guess, bounds, n_starts, exploitation_weight

    def _calculate_adaptive_scaling(self, params):
        """
        Calculate adaptive scaling factors based on current parameter values.
        """
        return [10.0**np.floor(np.log10(abs(p))) for p in params]
    
    def fit_to_data(self, data: pd.DataFrame, track_fitting_progress: bool = False,
                    fitting_settings: Optional[dict] = None) -> dict:
        """
        Fit model parameters to experimental data.
        
        Parameters
        ----------

        data : pd.DataFrame
            Experimental data (with appropriate columns depending on mode).
        track_fitting_progress : bool, optional
            Whether to track optimisation progress.
        fitting_settings : dict, optional
            Settings dictionary that may include:
            - mode: "d1" (fit only D1_prime) or "both" (fit both D1_prime and DT_0).
            - initial_guess: float (if mode "d1") or tuple (if mode "both")
            - bounds: tuple (if mode "d1") or tuple of tuples (if mode "both")
            - n_starts: number of multi-starts (default 1)
            - exploitation_weight: balance between exploitation (1.0) and exploration (0.0)
            - track_fitting_progress: whether to track optimisation progress
        
        Returns
        -------

        dict
            Dictionary containing fitted parameters and RMSE.
        """
        # Determine mode (default to "D1")
        mode = fitting_settings.get("mode", "D1") if fitting_settings else "D1"
        
        # Process fitting settings into initial_guess, bounds, and n_starts
        initial_guess, bounds, n_starts, exploitation_weight = self._process_fitting_settings(fitting_settings)
        
        if mode == "D1":
            required_cols = ['tau', 'normalised_flux']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Data is missing required columns: {missing_cols}")
            return self._fit_D1prime(data, initial_guess=initial_guess, bounds=bounds, n_starts=n_starts,
                                      track_fitting_progress=track_fitting_progress)
        elif mode == "both":
            # Check required columns for fitting both parameters
            required_cols = ['time', 'normalised_flux']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Data is missing required columns: {missing_cols}")
            return self._fit_D1prime_DT0(data, initial_guess=initial_guess, bounds=bounds, n_starts=n_starts, 
                                        exploitation_weight=exploitation_weight)
        else:
            raise ValueError("Invalid mode. Choose 'D1' or 'both'.")
        
    def _fit_D1prime(self, data: pd.DataFrame, initial_guess=5.0, bounds=(1.01, 100), n_starts=1,
                      exploitation_weight: float=0.7, track_fitting_progress: bool = True) -> dict:
        """
        Helper function to optimize only D1_prime.
        
        Parameters
        ----------

        data : pd.DataFrame
            Experimental data.
        initial_guess : float
            Initial guess for D1_prime.
        bounds : tuple
            Bounds for D1_prime.
        n_starts : int
            Number of multi-start optimisations.
        exploitation_weight : float
            Balance between exploitation (1.0) and exploration (0.0)
        track_fitting_progress : bool
            Whether to track optimisation progress.
        
        Returns
        -------

        dict
            Dictionary containing fitted D1_prime and RMSE.
        """
        if not isinstance(n_starts, int) or n_starts < 1:
            raise ValueError("n_starts must be an integer greater than or equal to 1")
        
        best_result = None
        best_rmse = np.inf
        callback_instance = OptimisationCallback(param_names=["D1_prime"]) if track_fitting_progress else None
        successful_params = []
        last_rmse = [float('inf')]
        
        def objective(params):
            D1_prime = params
            _, flux_df = self._solve_pde(
                L=self.params.transport.thickness,
                D1_prime=D1_prime,
                DT_0=self.params.transport.DT_0,
                T=data['time'].max(),
                X=1.0,
                dx=0.005,
                rel_tol=1e-6,  # Use moderate tolerance for fitting
                track_solving_progress=False
            )
            model_norm_flux = np.interp(data['tau'], flux_df['tau'], flux_df['normalised_flux'])
            rmse = np.sqrt(np.mean((data['normalised_flux'] - model_norm_flux) ** 2))
            last_rmse[0] = rmse
            return rmse

        bounds_list = [bounds]
        
        for i in range(n_starts):
            # Determine starting point using exploration vs exploitation
            if i == 0 or not successful_params or np.random.random() > exploitation_weight:
                # Exploration: Use initial guess or random values
                if i == 0:
                    current_guess = initial_guess
                else:
                    current_guess = np.random.uniform(bounds[0], bounds[1])
            else:
                # Exploitation: Sample near successful previous results
                base_param = successful_params[np.random.randint(len(successful_params))]
                # Add random perturbation (10% of parameter range)
                perturbation = np.random.normal(0, 0.1 * (bounds[1] - bounds[0]))
                current_guess = np.clip(base_param + perturbation, bounds[0], bounds[1])
            
            x0 = [current_guess]
            
            def local_callback(xk):
                if callback_instance is not None:
                    callback_instance(xk, last_rmse[0])
            
            result = minimize(
                lambda x: objective(x[0]),
                x0=x0,
                method='L-BFGS-B',
                bounds=bounds_list,
                callback=local_callback if track_fitting_progress else None
            )
            
            if result.success:
                successful_params.append(result.x[0])
            
            if result.fun < best_rmse:
                best_rmse = result.fun
                best_result = result

        if callback_instance is not None:
            callback_instance.close()

        return {
            'D1_prime': best_result.x[0],
            'rmse': best_rmse,
            'optimisation_result': best_result,
            'optimisation_history': callback_instance.history if callback_instance is not None else [],
            'n_successful': len(successful_params)
        }

    def _fit_D1prime_DT0(self, data: pd.DataFrame, 
                         initial_guess: Tuple=(5.0, 1e-7),
                         bounds: Tuple=((1.01, 100), (1e-8, 1e-5)),
                         n_starts: int=1,
                         exploitation_weight: float=0.7) -> dict:
        """
        Multi-stage parameter fitting strategy using solve_ivp with progressively finer tolerances
        and multi-start optimization for robustness.
        
        Parameters
        ----------
        data : pd.DataFrame
            Experimental data with 'time' and 'normalised_flux' columns
        initial_guess : Tuple
            Initial guess for (D1_prime, DT_0)
        bounds : Tuple
            Bounds for (D1_prime, DT_0)
        n_starts : int
            Number of multi-start optimizations
        exploitation_weight : float
            Balance between exploitation (1.0) and exploration (0.0)
            
        Returns
        -------
        dict
            Dictionary containing fitted parameters and optimization results
        """
        if not isinstance(n_starts, int) or n_starts < 1:
            raise ValueError("n_starts must be an integer greater than or equal to 1")
        
        stages = [
            # (dx, rel_tol)
            (0.02, 1e-6),   # Stage 1: Very coarse
            (0.01, 1e-7),   # Stage 2: Medium
            (0.005, 1e-8)   # Stage 3: Fine
        ]
        
        best_result = None
        best_rmse = np.inf
        successful_params = []
        last_rmse = [float('inf')]
        
        def objective(params, dx, tol, stage):
            D1_prime, DT_0 = params
            try:
                _, flux_df = self._solve_pde(
                    L=self.params.transport.thickness,
                    D1_prime=D1_prime, 
                    DT_0=DT_0,
                    T=data['time'].max(), 
                    X=1.0,
                    dx=dx, 
                    rel_tol=tol,
                    dt_init=0.001 if stage == 1 else 0.0001,  # Smaller initial step for finer stages
                    track_solving_progress=False
                )
                model_flux = np.interp(data['time'], flux_df['time'], 
                                     flux_df['normalised_flux'])
                rmse = np.sqrt(np.mean((data['normalised_flux'] - model_flux) ** 2))
                last_rmse[0] = rmse
                return rmse
            except Exception as e:
                # Return large error if simulation fails
                return 1e6
        
        # Perform multi-start optimization
        for i in range(n_starts):
            # Determine starting point using exploration vs exploitation
            if i == 0 or not successful_params or np.random.random() > exploitation_weight:
                # Exploration: Use initial guess or random values
                if i == 0:
                    current_guess = initial_guess
                else:
                    # Generate random starting point within bounds
                    current_guess = (
                        np.random.uniform(bounds[0][0], bounds[0][1]),  # D1_prime
                        np.random.uniform(bounds[1][0], bounds[1][1])   # DT_0
                    )
            else:
                # Exploitation: Sample near successful previous results
                base_params = successful_params[np.random.randint(len(successful_params))]
                # Add random perturbation (10% of parameter range)
                d1_range = bounds[0][1] - bounds[0][0]
                dt_range = bounds[1][1] - bounds[1][0]
                perturbation = (
                    np.random.normal(0, 0.1 * d1_range),
                    np.random.normal(0, 0.1 * dt_range)
                )
                current_guess = (
                    np.clip(base_params[0] + perturbation[0], bounds[0][0], bounds[0][1]),
                    np.clip(base_params[1] + perturbation[1], bounds[1][0], bounds[1][1])
                )
            
            # Multi-stage optimization for current starting point
            stage_guess = current_guess
            stage_best_result = None
            stage_best_rmse = np.inf
            
            for stage, (dx, tol) in enumerate(stages, 1):
                def stage_objective(params):
                    return objective(params, dx, tol, stage)
                
                # Optimize with current stage settings
                result = minimize(
                    stage_objective,
                    x0=stage_guess,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'ftol': tol,
                        'gtol': tol,
                        'maxiter': 50 if stage < 3 else 100
                    }
                )
                
                # Update stage best result if improved
                if result.success and result.fun < stage_best_rmse:
                    stage_best_rmse = result.fun
                    stage_best_result = result
                    
                # Use current result as initial guess for next stage
                stage_guess = result.x if result.success else stage_guess
            
            # Track successful parameters for exploitation
            if stage_best_result is not None and stage_best_result.success:
                successful_params.append(stage_best_result.x)
            
            # Update global best result
            if stage_best_result is not None and stage_best_rmse < best_rmse:
                best_rmse = stage_best_rmse
                best_result = stage_best_result
        
        return {
            'D1_prime': best_result.x[0] if best_result is not None else initial_guess[0],
            'DT_0': best_result.x[1] if best_result is not None else initial_guess[1],
            'rmse': best_rmse,
            'optimisation_result': best_result,
            'n_successful': len(successful_params)
        }