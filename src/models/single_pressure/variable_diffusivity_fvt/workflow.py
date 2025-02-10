import os
from typing import Dict, Optional, Tuple, Any
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from ....utils.data_processing import preprocess_data
from ..variable_diffusivity_fvt import FVTModel
import time
from ..variable_diffusivity_fvt.plotting import (
    plot_diffusivity_profile,
    plot_diffusivity_location_profile,
    plot_norm_flux_over_tau,
    plot_norm_flux_over_time
)

# Default settings dictionaries
DEFAULT_SIMULATION_PARAMS = {
    'T': 10e3,  # total time [s]
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

# New default fitting settings constant
DEFAULT_FITTING_SETTINGS = {
    'mode': 'd1',         # 'd1' or 'both'
    'initial_guess': 5.0,   # 5.0 or (5.0, 1e-7) when mode is 'both'
    'bounds': (1.001, 20),  # or ((1.001, 20), (1e-7, 1e-5))
    'n_starts': 1
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
        'T': 10e3,  # total time [s]
        'dx': 0.002,   # spatial step [adim]
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
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5)) 1x3 plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))    # 2x2 plot
    
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
    
    plot_norm_flux_over_time(
        flux_data=flux_df,
        experimental_data=experimental_data,
        ax=ax4,
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
    DT_0: float,
    D1_prime: float,
    fitting_settings: Optional[dict] = None,  # New fitting_settings argument
    stabilisation_threshold: Optional[float] = 0.003,    # New optional parameter
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
    DT_0 : float
        D0(T) parameter
    D1_prime : float
        D1' parameter
    fitting_settings : dict, optional
        Settings dictionary that may include:
        - mode: "d1" (fit only D1_prime) or "both" (fit both D1_prime and DT_0)
        - initial_guess: float (if mode "d1") or tuple (if mode "both")
        - bounds: tuple (if mode "d1") or tuple of tuples (if mode "both")
        - n_starts: number of multi-starts (default 1)
    output_settings : dict, optional
        Dictionary containing output settings (see defaults above)
    stabilisation_threshold : float, optional
        Threshold used in the data processing to determine stabilisation time (default: 0.003)
    
    Returns
    -------
    model : FVTModel
        Fitted FVT model
    fit_results : dict
        Fitting results including parameters and RMSE
    figures : dict
        Dictionary containing figure objects for plots
    """
    # Initialize model with initial parameters
    model = FVTModel.from_parameters(
        pressure=pressure,
        temperature=temperature,
        thickness=thickness,
        diameter=diameter,
        flowrate=flowrate,
        DT_0=DT_0,  # placeholder value
        D1_prime=D1_prime,   # placeholder value
    )
    
    # Load experimental data
    exp_data = pd.read_excel(data_path)
    
    # Preprocess data and calculate tau using provided stabilisation_threshold
    processed_exp_data = preprocess_data(
        exp_data,
        thickness=model.params.transport.thickness,
        diameter=model.params.transport.diameter,
        flowrate=model.params.transport.flowrate,
        temperature=model.params.base.temperature,
        stabilisation_threshold=stabilisation_threshold,
        truncate_at_stabilisation=True,
    )
    
    # Create 'tau' column
    processed_exp_data['tau'] = model.params.transport.DT_0 * processed_exp_data['time'] / model.params.transport.thickness**2
    
    # Downsample to 1000 points for faster optimization
    if len(processed_exp_data) > 1000:
        n = len(processed_exp_data) // 1000
        processed_exp_data = processed_exp_data.iloc[::n].reset_index(drop=True)
    
    # Merge default fitting settings using DEFAULT_FITTING_SETTINGS
    final_fitting_settings = {**DEFAULT_FITTING_SETTINGS, **(fitting_settings or {})}
    
    # Fit model to data with tracking
    fit_results = model.fit_to_data(
        data=processed_exp_data,
        fitting_settings=final_fitting_settings,
        track_fitting_progress=True
    )
    time.sleep(0.5)
    print("Fitting completed. Fitting Results:")
    for key, value in fit_results.items():
        print(f"{key}: {value}")
    
    # Run simulation with fitted parameters
    model, Dprime_df, flux_df, figures = manual_workflow(
        pressure=pressure,
        temperature=temperature,
        thickness=thickness,
        diameter=diameter,
        flowrate=flowrate,
        DT_0=DT_0,
        D1_prime=fit_results['D1_prime'],
        experimental_data=processed_exp_data,
        simulation_params={
            'T': processed_exp_data['time'].max(),
            'X': 1.0,
            'dx': 0.005,
            },
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

