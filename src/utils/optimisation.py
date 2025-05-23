from typing import Dict, List, Any
import numpy as np
# from tqdm.notebook import tqdm
from tqdm import tqdm, tqdm_notebook
import sys

class OptimisationCallback:
    """Callback class for optimization progress tracking"""
    def __init__(self, param_names: List[str]):
        self.param_names = param_names
        self.history: List[Dict[str, Any]] = []
        # self.pbar = tqdm(desc="Fitting (multi-start)", unit="iter")   # tqdm from tqdlm.notebook, hence not working in terminal
        # Use tqdm_notebook in Jupyter Notebook, otherwise use tqdm
        if 'ipykernel' in sys.modules:
            self.pbar = tqdm_notebook(desc="Fitting", unit="iter")
        else:
            self.pbar = tqdm(desc="Fitting", unit="iter")        
        
    def __call__(self, xk: np.ndarray, rmse: float) -> None:
        """
        Called automatically after each iteration
        
        Parameters
        ----------
        xk : np.ndarray
            Current parameter values
        rmse : float
            Current RMSE value
        """
        if not hasattr(xk, '__iter__'):
            xk = [xk]
        
        # Create iteration data dictionary
        iteration_data = {
            'iteration': len(self.history),  # Current iteration number
            **dict(zip(self.param_names, xk)),  # Parameter names + values
            'rmse': rmse  # Root Mean Square Error
        }
        
        # Store in history
        self.history.append(iteration_data)
        
        # Update progress bar with parameters and RMSE
        self.pbar.update(1)
        
        # Create parameter display
        postfix = {name: f"{val:.2e}" for name, val in zip(self.param_names, xk)}
        postfix['rmse'] = f"{rmse:.2e}"
        self.pbar.set_postfix(**postfix, refresh=True)
    
    def close(self) -> None:
        """Clean up progress bar"""
        self.pbar.close()