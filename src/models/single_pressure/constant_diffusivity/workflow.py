import os
import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import time

from ....utils.data_processing import preprocess_data
from ....utils.dir_paths import safe_long_path
from .model import TimelagModel
from .plotting import (
    plot_timelag_analysis,
    plot_concentration_profile,
    plot_flux_over_time
)

def manual_workflow(
    pressure: float,
    temperature: float,
    thickness: float,
    diameter: float,
    flowrate: float,
    diffusivity: float,
    equilibrium_concentration: float,
    experimental_data: Optional[pd.DataFrame] = None,
    simulation_params: Dict = {
        'T': 10e3,  # total time [s]
        'dx': 0.002,   # spatial step [adim]
        'X': 1.0,      # normalized position
        'rel_tol': 1e-8,  # relative tolerance
        'atol': 1e-9    # absolute tolerance
    },
    output_settings: Dict[str, Any] = {
        'output_dir': None,
        'display_plots': True,
        'save_plots': True,
        'save_data': True,
        'plot_format': 'png',
        'data_format': 'csv'
    }
) -> Tuple[TimelagModel, pd.DataFrame, pd.DataFrame, Dict[str, plt.Figure]]:
    
    # Always create timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Setup output directories
    if output_settings.get('output_dir'):
        os.makedirs(output_settings['output_dir'], exist_ok=True)
    
    time.sleep(0.5)  # Small delay for better timing in outputs
    
    # Initialize model
    model = TimelagModel.from_parameters(
        pressure=pressure,
        temperature=temperature,
        thickness=thickness,
        diameter=diameter,
        flowrate=flowrate,
        diffusivity=diffusivity,
        equilibrium_concentration=equilibrium_concentration,
    )
    
    # Solve diffusion PDE using solve_ivp method (more robust than finite difference)
    # dt and dx parameters are converted to appropriate grid sizing for solve_ivp
    conc_profile, flux_data = model.solve_pde(
        D=diffusivity,
        C_eq=equilibrium_concentration,
        L=thickness,
        T=simulation_params['T'],
        dt=simulation_params['dt'],  # Larger time step is okay with solve_ivp's adaptive timestepping
        dx=simulation_params['dx']  # Use relative spatial resolution
    )
    
    # Create plots
    figures = {}
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)    # 2x1 plot

    plot_concentration_profile(
        conc_profile,
        ax=ax1,
        display=False,
    )
    
    plot_flux_over_time(
        flux_data,
        experimental_data=experimental_data,
        ax=ax2,
        display=False
    )
    
    figures['combined'] = fig
    
    # Save outputs with timestamps
    if output_settings['save_plots'] and output_settings.get('output_dir'):
        plot_path = os.path.join(
            output_settings['output_dir'], 
            f'constant_diffusivity_analysis_summary_{timestamp}.{output_settings["plot_format"]}'
        )
        plot_path = safe_long_path(plot_path)
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')

    # Export data
    if output_settings['save_data'] and output_settings.get('output_dir'):
        data_format = output_settings.get('data_format', 'csv')
        if data_format == 'csv':
            if experimental_data is not None:
                # Experimental data
                experimental_data.to_csv(
                    safe_long_path(os.path.join(output_settings['output_dir'], f'experimental_data_{timestamp}.csv')),
                    index=False
                )

            # Concentration profile
            conc_profile.to_csv(
                safe_long_path(os.path.join(output_settings['output_dir'], f'concentration_profile_{timestamp}.csv')),
                index=False
            )
            
            # Flux data
            flux_data.to_csv(
                safe_long_path(os.path.join(output_settings['output_dir'], f'flux_data_{timestamp}.csv')),
                index=False
            )
    
    return model, conc_profile, flux_data, figures

def data_fitting_workflow(
    file_path: str,
    thickness: float,
    diameter: float,
    pressure: float,
    temperature: float,
    flowrate: float,
    stabilisation_threshold: Optional[float] = 0.003,
    output_settings: Dict[str, Any] = {
        'output_dir': None,
        'display_plots': True,
        'save_plots': True,
        'save_data': True,
        'plot_format': 'png',
        'data_format': 'csv'
    }
) -> Dict[str, Any]:
    """Execute time-lag analysis workflow"""
    
    # Setup output directories
    output_dir = output_settings.get('output_dir')
    if output_dir and (output_settings.get('save_plots') or output_settings.get('save_data')):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        timestamp = None

    # Create model and fit
    model = TimelagModel.from_parameters(
        pressure=pressure,
        temperature=temperature,
        thickness=thickness,
        diameter=diameter,
        flowrate=flowrate
    )
    
    # Load data
    exp_data = pd.read_excel(file_path)
    
    # Process data
    processed_exp_data = preprocess_data(
        exp_data,
        thickness=model.params.transport.thickness,
        diameter=model.params.transport.diameter,
        flowrate=model.params.transport.flowrate,
        temperature=model.params.base.temperature,
        stabilisation_threshold=stabilisation_threshold,
        truncate_at_stabilisation=True,
    )
    
    model.fit_to_data(processed_exp_data)
    
    # Call manual_workflow to run the model with the fitted parameters
    _, conc_profile, flux_data, figures = manual_workflow(
        pressure=model.params.base.pressure,
        temperature=model.params.base.temperature,
        thickness=model.params.transport.thickness,
        diameter=model.params.transport.diameter,
        flowrate=model.params.transport.flowrate,
        diffusivity=model.results.get('diffusivity', None),
        equilibrium_concentration=model.results.get('equilibrium_concentration', None),
        experimental_data=processed_exp_data,
        simulation_params={
            'T': max(processed_exp_data['time']),
            'dt': 5.0,
            'dx': thickness / 1000,
        },
        output_settings=output_settings
    )
    
    # Generate results dictionary
    results_dict = {
        'parameters': {
            'base': {
                'pressure': pressure,
                'temperature': temperature
            },
            'transport': {
                'thickness': thickness,
                'diameter': diameter,
                'flowrate': flowrate
            }
        },
        'results': model.results,
        'units': {
            'pressure': 'bar',
            'temperature': '°C',
            'thickness': 'cm',
            'diameter': 'cm',
            'flowrate': 'cm³(STP) s⁻¹',
            'diffusivity': 'cm² s⁻¹',
            'permeability': 'cm³(STP) cm⁻¹ s⁻¹ bar⁻¹',
            'solubility': 'cm³(STP) cm⁻³ bar⁻¹'
        }
    }

    # Link model results to results_dict (same object im memory)
    fit_results = model.results

    # Calculate RMSE if flux is available
    if 'flux' in processed_exp_data.columns:
        model_flux = np.interp(
            processed_exp_data['time'],
            flux_data['time'],
            flux_data['flux']
        )
        rmse = np.sqrt(np.mean((processed_exp_data['flux'] - model_flux) ** 2))
        fit_results['rmse'] = rmse
    
    # Generate plots
    plot_timelag_analysis(
        model, 
        processed_exp_data,
        save_path=safe_long_path(os.path.join(
            output_dir, f'timelag_analysis_{timestamp}.{output_settings["plot_format"]}')),
        display=output_settings.get('display_plots', True),
    )
    
    # Save fitting results with timestamp
    if output_settings['save_data'] and output_settings.get('output_dir'):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_path = os.path.join(
            output_settings['output_dir'], 
            f'fitting_results_{timestamp}.{output_settings["data_format"]}'
        )
        results_path = safe_long_path(results_path)
        if output_settings['data_format'] == 'csv':
            pd.DataFrame([{k: v for k, v in fit_results.items()}]
                        ).to_csv(results_path, index=False)
            
    return model, results_dict, figures, conc_profile, flux_data, processed_exp_data