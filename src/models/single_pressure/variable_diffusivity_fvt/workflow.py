import os
from typing import Dict, Optional, Tuple, Any
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from ..variable_diffusivity_fvt import FVTModel
from ..variable_diffusivity_fvt.plotting import (
    plot_diffusivity_profile,
    plot_diffusivity_location_profile,
    plot_norm_flux_over_tau,
)

# Default settings dictionaries
DEFAULT_SIMULATION_PARAMS = {
    'T': 100000,  # total time [s]
    'dt': 1.0,    # time step [s]
    'dx': 0.01,   # spatial step [adim]
    'X': 1.0      # normalized position
}

DEFAULT_OUTPUT_SETTINGS = {
    'output_dir': None,
    'display_plots': True,
    'save_plots': True,
    'save_data': True,
    'plot_format': 'png',
    'data_format': 'csv'
}

def manual_workflow(
    pressure: float,
    temperature: float,
    thickness: float,
    diameter: float,
    flowrate: float,
    D1_prime: float,
    DT_0: float,
    experimental_data: Optional[pd.DataFrame] = None,
    simulation_params: Dict = {
        'T': 100000,  # total time [s]
        'dt': 1.0,    # time step [s]
        'dx': 0.01,   # spatial step [adim]
        'X': 1.0      # normalized position
    },
    output_settings: Dict[str, Any] = {
        'output_dir': None,
        'display_plots': True,
        'save_plots': True,
        'save_data': True,
        'plot_format': 'png',
        'data_format': 'csv'
    }
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
    DT_0 : float
        D0(T) parameter
    simulation_params : dict, optional
        Dictionary containing simulation parameters:
        - T: total time [s]
        - dt: time step [s]
        - dx: spatial step [adim]
    experimental_data : pd.DataFrame, optional
        Experimental flux data for comparison
    output_settings : dict, optional
        Dictionary containing output settings:
        - output_dir: Directory to save outputs (default: None)
        - display_plots: Whether to display plots (default: True)
        - save_plots: Whether to save plots (default: True)
        - save_data: Whether to save data (default: True)
        - plot_format: Format for saving plots (default: 'png')
        - data_format: Format for saving data (default: 'csv')
    
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
    # Use default settings with any overrides
    sim_params = {**DEFAULT_SIMULATION_PARAMS, **(simulation_params or {})}
    output_settings = {**DEFAULT_OUTPUT_SETTINGS, **(output_settings or {})}
    
    # Setup output directories
    if output_settings.get('output_dir'):
        os.makedirs(output_settings['output_dir'], exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Initialize model
    model = FVTModel.from_parameters(
        pressure=pressure,
        temperature=temperature,
        thickness=thickness,
        diameter=diameter,
        flowrate=flowrate,
        D1_prime=D1_prime,
        DT_0=DT_0
    )
    
    # Solve PDE with updated method
    Dprime_df, flux_df = model.solve_pde(simulation_params=simulation_params)
    
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
        T=sim_params['T'],
        ax=ax2,
        display=False
    )
    
    # Plot flux evolution
    plot_norm_flux_over_tau(
        flux_data=flux_df,
        experimental_data=experimental_data,
        ax=ax3,
        display=False
    )
    
    figures['combined'] = fig
    
    # Save outputs with timestamps
    if output_settings['save_plots'] and output_settings.get('output_dir'):
        plot_path = os.path.join(
            output_settings['output_dir'], 
            f'fvt_analysis_summary_{timestamp}.{output_settings["plot_format"]}'
        )
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    if output_settings['save_data'] and output_settings.get('output_dir'):
        if output_settings['data_format'] == 'csv':
            Dprime_df.to_csv(os.path.join(
                output_settings['output_dir'], 
                f'diffusivity_profile_{timestamp}.csv'
            ))
            flux_df.to_csv(os.path.join(
                output_settings['output_dir'], 
                f'flux_evolution_{timestamp}.csv'
            ))
        elif output_settings['data_format'] == 'excel':
            with pd.ExcelWriter(os.path.join(
                output_settings['output_dir'], 
                f'fvt_results_{timestamp}.xlsx'
            )) as writer:
                Dprime_df.to_excel(writer, sheet_name='diffusivity_profile')
                flux_df.to_excel(writer, sheet_name='flux_evolution')
    
    if output_settings['display_plots']:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)
    
    return model, Dprime_df, flux_df, figures

def data_fitting_workflow(
    data_path: str,
    pressure: float,
    temperature: float,
    thickness: float,
    diameter: float,
    flowrate: float,
    initial_guess: Dict[str, float] = {
        'D1_prime': 2.38,
        'DT_0': 2.87e-7
    },
    simulation_params: Dict = {
        'T': 100000,  # total time [s]
        'dt': 1.0,    # time step [s]
        'dx': 0.01,   # spatial step [adim]
        'X': 1.0      # normalized position
    },
    output_settings: Dict[str, Any] = {
        'output_dir': None,
        'display_plots': True,
        'save_plots': True,
        'save_data': True,
        'plot_format': 'png',
        'data_format': 'csv'
    }
) -> Tuple[FVTModel, Dict[str, float], Dict[str, plt.Figure]]:
    """
    Run complete FVT data fitting workflow including model fitting and result visualization
    
    Parameters
    ----------
    data_path : str
        Path to experimental data file
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
    initial_guess : dict, optional
        Initial guess for fitting parameters:
        - D1_prime: normalized diffusivity at x=0
        - DT_0: temperature-dependent diffusivity
    simulation_params : dict, optional
        Dictionary containing simulation parameters
    output_settings : dict, optional
        Dictionary containing output settings:
        - output_dir: Directory to save outputs (default: None)
        - display_plots: Whether to display plots (default: True)
        - save_plots: Whether to save plots (default: True)
        - save_data: Whether to save data (default: True)
        - plot_format: Format for saving plots (default: 'png')
        - data_format: Format for saving data (default: 'csv')
    
    Returns
    -------
    model : FVTModel
        Fitted FVT model
    fit_results : dict
        Fitting results including parameters and RMSE
    figures : dict
        Dictionary containing figure objects for plots
    """
    # Load experimental data
    experimental_data = pd.read_excel(data_path)
    
    # Initialize model with initial parameters
    model = FVTModel.from_parameters(
        pressure=pressure,
        temperature=temperature,
        thickness=thickness,
        diameter=diameter,
        flowrate=flowrate,  # Add flowrate parameter
        D1_prime=initial_guess['D1_prime'],
        DT_0=initial_guess['DT_0']
    )
    
    # Fit model to data
    fit_results = model.fit_to_data(
        data=experimental_data,
        simulation_params=simulation_params
    )
    
    # Run simulation with fitted parameters
    model, Dprime_df, flux_df, figures = manual_workflow(
        pressure=pressure,
        temperature=temperature,
        thickness=thickness,
        diameter=diameter,
        D1_prime=fit_results['D1_prime'],
        DT_0=fit_results['DT_0'],
        experimental_data=experimental_data,
        simulation_params=simulation_params,
        output_settings=output_settings
    )
    
    # Save fitting results with timestamp
    if output_settings['save_data'] and output_settings.get('output_dir'):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_path = os.path.join(
            output_settings['output_dir'], 
            f'fitting_results_{timestamp}.{output_settings["data_format"]}'
        )
        if output_settings['data_format'] == 'csv':
            pd.DataFrame([fit_results]).to_csv(results_path, index=False)
        else:
            with open(results_path, 'w') as f:
                f.write("FVT Model Fitting Results\n")
                f.write("========================\n\n")
                f.write(f"D1_prime: {fit_results['D1_prime']:.4e}\n")
                f.write(f"DT_0: {fit_results['DT_0']:.4e}\n")
                f.write(f"RMSE: {fit_results['rmse']:.4e}\n")
    
    return model, fit_results, figures

