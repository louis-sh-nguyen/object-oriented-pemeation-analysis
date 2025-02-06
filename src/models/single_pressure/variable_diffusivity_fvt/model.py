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
import time

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
                lapl = (D_new[i+1] - 2.0 * D_new[i] + D_new[i-1]) / dx2 # Laplacian for second-derivative
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
                # Laplacian for second-derivative
                lapl = (D_new[i+1] - 2.0 * D_new[i] + D_new[i-1]) / dx2 
                #  Diagonal of Jacobian, by taking the partial derivative of the residual R[i] with respect to D[i]
                J_diag = (1.0 / dt) - K * (lapl + (-2.0 * D_new[i]) / dx2)  # The -2.0 factor comes from differentiating the centered second‐difference approximation (i.e., D[i+1] - 2.0*D[i] + D[i-1]) with respect to D[i].
                if J_diag == 0.0:
                    J_diag = 1e-8   # Avoid division by zero
                D_new[i] = D_new[i] - relax * (((D_new[i] - D_old[i]) / dt) - K * D_new[i] * lapl) / J_diag # implicit update

            D_new[0] = D1_prime
            D_new[Nx - 1] = D2_prime
        return D_new, converged

    def _solve_pde(self, D1_prime, DT_0, T, X, L, dx, 
                  D2_prime=1.0, rel_tol=1e-8, max_iter=100, relax=0.8,
                  dt_init=0.0005, dt_target=10, dt_min=1e-6, dt_ramp_factor=1.1,
                  track_solving_progress=True):
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

        if track_solving_progress:
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
            if track_solving_progress:
                pbar.update(dt) # increments the progress by an amount dt
        if track_solving_progress:
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
    
    def _process_fitting_settings(self, mode: str, fitting_settings: Optional[dict]) -> tuple:
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
        n_starts = 1
        if fitting_settings:
            n_starts = fitting_settings.get("n_starts", 1)
        if mode == "both":
            default_init = (self.params.transport.D1_prime, self.params.transport.DT_0) if hasattr(self, "params") else (5.0, 1e-6)
            initial_guess = fitting_settings.get("initial_guess", default_init) if fitting_settings else default_init
            default_bounds = ((1.01, 100), (1e-8, 1e-5))
            bounds = fitting_settings.get("bounds", default_bounds) if fitting_settings else default_bounds
        else:  # mode == "d1"
            default_init = self.params.transport.D1_prime if hasattr(self, "params") else 5.0
            initial_guess = fitting_settings.get("initial_guess", default_init) if fitting_settings else default_init
            default_bounds = (1.01, 100)
            bounds = fitting_settings.get("bounds", default_bounds) if fitting_settings else default_bounds
        return initial_guess, bounds, n_starts

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
        
        Returns
        -------
        dict
            Dictionary containing fitted parameters and RMSE.
        """
        # Determine mode (default to "D1")
        mode = fitting_settings.get("mode", "D1") if fitting_settings else "D1"
        # Process fitting settings into initial_guess, bounds, and n_starts.
        initial_guess, bounds, n_starts = self._process_fitting_settings(mode, fitting_settings)
        
        if mode == "both":
            # Check required columns for fitting both parameters.
            required_cols = ['time', 'tau', 'normalised_flux']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Data is missing required columns: {missing_cols}")
            return self._fit_D1prime_DT0(data, initial_guess=initial_guess, bounds=bounds, n_starts=n_starts,
                                         track_fitting_progress=track_fitting_progress)
        else:
            # Default mode: fit only D1_prime.
            required_cols = ['tau', 'normalised_flux']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Data is missing required columns: {missing_cols}")
            return self._fit_D1_prime(data, initial_guess=initial_guess, bounds=bounds, n_starts=n_starts,
                                      track_fitting_progress=track_fitting_progress)

    def _fit_D1_prime(self, data: pd.DataFrame, initial_guess=5.0, bounds=(1.01, 100), n_starts=1,
                      track_fitting_progress: bool = True) -> dict:
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
        track_progress : bool
            Whether to track optimisation progress.
        
        Returns
        -------
        dict
            Dictionary containing fitted D1_prime and RMSE.
        """
            # Validate that n_starts is an integer greater than or equal to 1
        if not isinstance(n_starts, int) or n_starts < 1:
            raise ValueError("n_starts must be an integer greater than or equal to 1")
        
        best_result = None
        best_fun = np.inf
        best_rmse = np.inf

        # Create callback only if tracking progress.
        callback_instance = OptimisationCallback(param_names=["D1_prime"]) if track_fitting_progress else None
        last_rmse = [float('inf')]
        
        def objective(params):
            D1_prime = params
            # Calculate flux using the current D1_prime
            _, flux_df = self._solve_pde(
                L=self.params.transport.thickness,
                D1_prime=D1_prime,
                DT_0=self.params.transport.DT_0,
                T=data['time'].max(),
                X=1.0,
                dx=0.005,
                track_solving_progress=False
            )
            # Interpolate model normalized flux to data tau points
            model_norm_flux = np.interp(data['tau'], flux_df['tau'], flux_df['normalised_flux'])
            rmse = np.sqrt(np.mean((data['normalised_flux'] - model_norm_flux) ** 2))
            last_rmse[0] = rmse
            return rmse

        # Prepare bounds in required format: [(low, high)]
        bounds_list = [bounds]
        
        for i in range(n_starts):
            # Use the provided initial guess for the first start; for subsequent starts use a random candidate within bounds
            if i == 0:
                x0 = [initial_guess]
            else:
                low, high = bounds
                candidate = np.random.uniform(low, high)
                x0 = [candidate]
            
            # Define a local callback that sends the current parameter vector and last RMSE to our callback_instance
            def local_callback(xk):
                if callback_instance is not None:
                    # Pass current xk and the last computed rmse
                    callback_instance(xk, last_rmse[0])
            
            # Run minimization with L-BFGS-B for this starting point
            result = minimize(
                lambda x: objective(x[0]),
                x0=x0,
                method='L-BFGS-B',
                bounds=bounds_list,
                callback=local_callback
            )
            if result.fun < best_fun:
                best_fun = result.fun
                best_result = result
                best_rmse = result.fun
            
            # Print progress message after each run
            time.sleep(0.5)
            print(f"Optimization run {i+1}: D1_prime = {result.x[0]}, RMSE = {result.fun}")
        
        if callback_instance is not None:
            callback_instance.close()

        best_params = {
            'D1_prime': best_result.x[0],
            'rmse': best_rmse,
            'optimisation_result': best_result,
            'optimisation_history': callback_instance.history if callback_instance is not None else []
        }
        # print(f"Best optimization result: {best_result}")
        
        return best_params

    def _fit_D1prime_DT0(self, data: pd.DataFrame, initial_guess=(5.0, 1e-7),
                          bounds=((1.01, 100), (1e-8, 1e-5)), n_starts=1,
                          track_fitting_progress: bool = False) -> dict:
        """
        Helper function to optimize both D1_prime and DT_0.
        
        Parameters
        ----------
        data : pd.DataFrame
            Experimental data.
        initial_guess : tuple
            Initial guess for (D1_prime, DT_0).
        bounds : tuple of tuples
            Bounds for (D1_prime, DT_0).
        n_starts : int
            Number of multi-start optimisations.
        track_progress : bool
            Whether to track optimisation progress.
        
        Returns
        -------
        dict
            Dictionary containing fitted D1_prime, DT_0 and RMSE.
        """
        # [Implementation for multi-start optimisation for both parameters]
        # ... (Your existing code using initial_guess, bounds, and n_starts)
        pass  # Replace with your actual implementation