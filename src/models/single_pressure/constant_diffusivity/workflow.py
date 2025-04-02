import os
import json
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime

from .model import TimelagModel
from .plotting import (
    plot_timelag_analysis,
    plot_concentration_profile,
    plot_flux_over_time
)

def time_lag_analysis_workflow(
    file_path: str,
    thickness: float,
    diameter: float,
    pressure: float,
    temperature: float,
    flowrate: Optional[float] = None,
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

    # Load data
    data = pd.read_excel(file_path)
    
    # Create model and fit
    model = TimelagModel.from_parameters(
        pressure=pressure,
        temperature=temperature,
        thickness=thickness,
        diameter=diameter,
        flowrate=flowrate
    )
    processed_data = model.fit_to_data(data)
    
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
    
    # Handle plotting and saving
    if output_dir and output_settings.get('save_plots'):
        plot_path = lambda name: os.path.join(
            output_dir, 'plots', f'{name}_{timestamp}.{output_settings["plot_format"]}'
        )
    else:
        plot_path = lambda name: None

    # Generate plots
    plot_timelag_analysis(
        model, 
        processed_data,
        save_path=plot_path('timelag_analysis'),
        display=output_settings.get('display_plots', True)
    )
    
    if model.results.get('diffusivity') and model.results.get('equilibrium_concentration'):
        # Solve diffusion PDE using solve_ivp method (more robust than finite difference)
        # dt and dx parameters are converted to appropriate grid sizing for solve_ivp
        conc_profile, flux_data = model.solve_pde(
            D=model.results['diffusivity'],
            C_eq=model.results['equilibrium_concentration'],
            L=model.params.transport.thickness,
            T=max(processed_data['time']),
            dt=5.0,  # Larger time step is okay with solve_ivp's adaptive timestepping
            dx=model.params.transport.thickness/100  # Use relative spatial resolution
        )
        
        plot_concentration_profile(
            conc_profile,
            save_path=plot_path('concentration_profile'),
            display=output_settings.get('display_plots', True)
        )
        
        plot_flux_over_time(
            flux_data,
            experimental_data=processed_data,
            save_path=plot_path('flux_evolution'),
            display=output_settings.get('display_plots', True)
        )

    # Save results
    if output_dir and output_settings.get('save_data'):
        data_format = output_settings.get('data_format', 'csv')
        if data_format == 'csv':
            processed_data.to_csv(
                os.path.join(output_dir, 'data', f'raw_data_{timestamp}.csv'),
                index=False
            )
            pd.DataFrame(results_dict['results']).to_csv(
                os.path.join(output_dir, 'data', f'processed_data_{timestamp}.csv')
            )
        elif data_format == 'json':
            with open(os.path.join(output_dir, 'results', f'model_results_{timestamp}.json'), 'w') as f:
                json.dump(results_dict, f, indent=4)

    return results_dict