import os
from typing import Dict, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from ..variable_diffusivity_fvt import FVTModel
from ..variable_diffusivity_fvt.plotting import (
    plot_diffusivity_profile,
    plot_diffusivity_location_profile,
    plot_flux_over_time
)

def fvt_workflow(
    pressure: float,
    temperature: float,
    thickness: float,
    diameter: float,
    D1_prime: float,
    D2_prime: float,
    D0_T: float,
    simulation_params: Optional[Dict] = None,
    experimental_data: Optional[pd.DataFrame] = None,
    output_dir: Optional[str] = None,
    display_plots: bool = True
) -> Tuple[FVTModel, pd.DataFrame, pd.DataFrame, Dict[str, plt.Figure]]:
    """
    Run complete FVT analysis workflow including model creation, solving and plotting
    
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
    D1_prime : float
        D1' parameter
    D2_prime : float
        D2' parameter
    D0_T : float
        D0(T) parameter
    simulation_params : dict, optional
        Dictionary containing simulation parameters:
        - T: total time [s]
        - dt: time step [s]
        - dx: spatial step [adim]
    experimental_data : pd.DataFrame, optional
        Experimental flux data for comparison
    output_dir : str, optional
        Directory to save output plots
    display_plots : bool, optional
        Whether to display the plots (default: True)
    
    Returns
    -------
    model : FVTModel
        Initialized and solved FVT model
    Dprime_df : pd.DataFrame
        Concentration profile results
    flux_df : pd.DataFrame
        Flux evolution results
    figures : dict
        Dictionary containing figure objects for each plot
    """
    # Set default simulation parameters if not provided
    if simulation_params is None:
        simulation_params = {
            'T': 100000,  # total time [s]
            'dt': 1.0,    # time step [s]
            'dx': 0.01,   # spatial step [adim]
            'X': 1.0      # normalized position
        }
    
    # Initialize model
    model = FVTModel.from_parameters(
        pressure=pressure,
        temperature=temperature,
        thickness=thickness,
        diameter=diameter,
        D1_prime=D1_prime,
        D2_prime=D2_prime,
        D0_T=D0_T
    )
    
    # Solve PDE
    Dprime_df, flux_df = model.solve_pde(
        D1_prime=D1_prime,
        D2_prime=D2_prime,
        D0_T=D0_T,
        T=simulation_params['T'],
        X=simulation_params['X'],
        L=thickness,
        dt=simulation_params['dt'],
        dx=simulation_params['dx']
    )
    
    # Create plots
    figures = {}
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot concentration profile evolution
    plot_diffusivity_profile(
        diffusivity_profile=Dprime_df,
        ax=ax1,
        display=False
    )
    
    # Plot concentration-location profiles
    plot_diffusivity_location_profile(
        diffusivity_profile=Dprime_df,
        L=thickness,
        T=simulation_params['T'],
        ax=ax2,
        display=False
    )
    
    # Plot flux evolution
    plot_flux_over_time(
        flux_data=flux_df,
        experimental_data=experimental_data,
        ax=ax3,
        display=False
    )
    
    figures['combined'] = fig
    
    # Save plots if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'fvt_analysis_summary.png'), 
                   dpi=300, bbox_inches='tight')
    
    if display_plots:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)
    
    return model, Dprime_df, flux_df, figures
