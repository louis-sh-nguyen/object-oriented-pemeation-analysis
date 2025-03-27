from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import minimize
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
        
        # print(f"Iteration {it}: norm_R = {norm_R:.3g}, norm_D = {norm_D:.3g}, norm_R/norm_D = {norm_R / norm_D:.3g}")
        
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
                  D2_prime=1.0, rel_tol=1e-8, max_iter=100, relax=0.8,
                  dt_init=0.0005, dt_target=10, dt_min=1e-6, dt_ramp_factor=1.1,
                  track_solving_progress=True, use_full_jacobian=True,
                  D0=None):  # Added D0 parameter to accept initial condition
        """
        Adaptive implicit PDE solver using Newton's method with adaptive dt.
        Can use either full Jacobian (default) or diagonal approximation.
        
        Parameters
        ----------
        D0 : ndarray, optional
            Initial condition for D profile. If None, defaults to flat profile.
        """
        Nx = int(X / dx) + 1
        x = np.linspace(0, X, Nx)
        K = DT_0 / (L * L)
        
        # Initial condition: D=1 everywhere except boundaries, or use provided D0
        if D0 is None:
            D0 = np.ones(Nx, dtype=np.float64)
            D0[0] = D1_prime     # Left boundary
            D0[-1] = D2_prime    # Right boundary
        else:
            # Ensure boundary conditions are properly set in provided D0
            D0[0] = D1_prime
            D0[-1] = D2_prime

        t_history = [0.0]
        D_history = [D0.copy()]
        flux_history = []
        current_t = 0.0
        dt = dt_init
        D_old = D0.copy()

        if track_solving_progress:
            pbar = tqdm(total=T, desc=f"Adaptive PDE Solve (D1'={D1_prime}, DTO={DT_0})", ncols=100)
            
        while current_t < T:
            accepted = False
            trial_dt = dt
            current_relax = relax  # start with the original relaxation
            
            while not accepted:
                D_new, converged = _newton_update_jit(D_history[-1], trial_dt, dx, K,
                                                    max_iter, D1_prime, D2_prime, current_relax, rel_tol,
                                                    use_full_jacobian)  # Added parameter
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
            
            # Calculate current flux for steady-state detection
            current_flux = (-(D_new[-1] - D_new[-2]) / dx) / (-(D2_prime - D1_prime) / X)
            flux_history.append(current_flux)
            
            # Adaptive time stepping logic
            if trial_dt < dt_target:
                new_dt = min(trial_dt * dt_ramp_factor, dt_target)
                # if new_dt > trial_dt:
                #     print("Increasing dt from", trial_dt, "to", new_dt)
                dt = new_dt
            else:
                dt = trial_dt
                
            D_old = D_new.copy()  # Keep last profile for comparison
            current_t += dt
            t_history.append(current_t)
            D_history.append(D_new.copy())
            
            if track_solving_progress:
                pbar.update(dt) # increments the progress by an amount dt

        if track_solving_progress:
            pbar.close()

        # Process results
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
        Solve PDE using model parameters.
        
        Parameters
        ----------
        simulation_params : dict, optional
            Dictionary containing simulation parameters:
            - T: total time [s]
            - dt: time step [s]
            - dx: spatial step [normalized]
            - X: normalized position
            - rel_tol: relative tolerance for convergence (default 1e-8)
            - max_iter: maximum iterations for Newton solver (default 100)
            - relax: relaxation parameter (default 0.8)
            - fitting_mode: set to True when called from fitting functions (default False)
            - use_full_jacobian: Whether to use the full Jacobian matrix (True) or diagonal approximation (False)
                               (default True for better convergence properties)
        
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
        max_iter = simulation_params.get('max_iter', 100)
        relax = simulation_params.get('relax', 0.8)
        fitting_mode = simulation_params.get('fitting_mode', False)
        use_full_jacobian = simulation_params.get('use_full_jacobian', True)
        
        # Single-stage solver with configurable Jacobian
        D_profile, flux = self._solve_pde(
            D1_prime=self.params.transport.D1_prime,
            DT_0=self.params.transport.DT_0,
            T=simulation_params.get('T', 10e3),
            X=simulation_params.get('X', 1.0),
            L=self.params.transport.thickness,
            dx=simulation_params.get('dx', 0.01),
            dt_init=simulation_params.get('dt', 0.001),
            dt_target=simulation_params.get('dt_target', 10),
            dt_min=simulation_params.get('dt_min', 1e-6),
            dt_ramp_factor=simulation_params.get('dt_ramp_factor', 1.1),
            rel_tol=rel_tol,
            max_iter=max_iter,
            relax=relax,
            use_full_jacobian=use_full_jacobian,
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
            return self._fit_D1prime_DT0(data, initial_guess=initial_guess, bounds=bounds, n_starts=n_starts)
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
                               n_starts: int=1) -> dict:
        """
        Multi-stage parameter fitting strategy:
        1. Coarse grid + diagonal Jacobian for rough fit
        2. Medium grid + diagonal Jacobian for refinement
        3. Fine grid + full Jacobian for final polish
        """
        stages = [
            # (dx, rel_tol, use_full_jacobian)
            (0.02, 1e-6, False),   # Stage 1: Very coarse
            (0.01, 1e-7, False),   # Stage 2: Medium
            (0.005, 1e-8, True)    # Stage 3: Fine
        ]
        
        best_result = None
        best_rmse = np.inf
        current_guess = initial_guess
        
        for stage, (dx, tol, use_full) in enumerate(stages, 1):
            def objective(params):
                D1_prime, DT_0 = params
                _, flux_df = self._solve_pde(
                    L=self.params.transport.thickness,
                    D1_prime=D1_prime, DT_0=DT_0,
                    T=data['time'].max(), X=1.0,
                    dx=dx, rel_tol=tol,
                    track_solving_progress=False,
                    use_full_jacobian=use_full  # Added parameter
                )
                model_flux = np.interp(data['time'], flux_df['time'], 
                                     flux_df['normalised_flux'])
                return np.sqrt(np.mean((data['normalised_flux'] - model_flux) ** 2))
            
            # Optimize with current settings
            result = minimize(
                objective,
                x0=current_guess,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'ftol': tol,
                    'gtol': tol,
                    'maxiter': 50 if stage < 3 else 100
                }
            )
            
            # Update best result if improved
            if result.fun < best_rmse:
                best_rmse = result.fun
                best_result = result
                
            # Use current result as initial guess for next stage
            current_guess = result.x
            
        return {
            'D1_prime': best_result.x[0],
            'DT_0': best_result.x[1],
            'rmse': best_rmse,
            'optimisation_result': best_result
        }