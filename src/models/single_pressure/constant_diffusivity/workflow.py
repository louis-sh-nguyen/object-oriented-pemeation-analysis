import os
import json
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
from ....utils.data_processing import preprocess_data
from ...parameters import BaseParameters
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
    flow_rate: float,
    pressure: float,
    temperature: float = 25.0,
    output_settings: Dict[str, Any] = {
        'output_dir': None,
        'display_plots': True,
        'save_plots': True,
        'save_data': True,
        'plot_format': 'png',  # 'png' or 'svg'
        'data_format': 'csv'   # 'csv' or 'json'
    }
) -> Dict[str, Any]:
    """Execute time-lag analysis workflow
    
    Parameters
    ----------
    # ...existing code...
    output_settings : Dict[str, Any], optional
        Dictionary controlling output options:
        - 'output_dir': Directory path for saving outputs (None for no saving)
        - 'display_plots': Display plots on screen
        - 'save_plots': Save plots to files (requires output_dir)
        - 'save_data': Save data files
        - 'plot_format': Format for saving plots ('png' or 'svg')
        - 'data_format': Format for saving data ('csv' or 'json')
    """
    output_dir = output_settings.get('output_dir')
    plot_format = output_settings.get('plot_format', 'png')
    data_format = output_settings.get('data_format', 'csv')
    
    # Setup output directories if saving is required
    if output_dir and any([output_settings.get('save_plots'),
                          output_settings.get('save_data')]):
        os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        timestamp = None

    # Load and process data
    data = pd.read_excel(file_path)
    
    # Preprocess data
    data = preprocess_data(data, thickness=thickness, diameter=diameter, flow_rate=flow_rate)
    
    # Create base parameters
    base_params = BaseParameters(
        thickness=thickness,
        diameter=diameter,
        flow_rate=flow_rate,
        pressure=pressure,
        temperature=temperature
    )
    
    # Create and fit model
    model = TimelagModel.from_data(data, base_params)
    
    # Generate results dictionary
    results_dict = {
        'parameters': {
            'thickness': thickness,
            'diameter': diameter,
            'flow_rate': flow_rate,
            'pressure': pressure,
            'temperature': temperature
        },
        'results': model.results,
        'units': {
            'thickness': 'cm',
            'diameter': 'cm',
            'flow_rate': 'cm³(STP) s⁻¹',
            'pressure': 'bar',
            'temperature': '°C',
            'diffusivity': 'cm² s⁻¹',
            'permeability': 'cm³(STP) cm⁻¹ s⁻¹ bar⁻¹',
            'solubility': 'cm³(STP) cm⁻³ bar⁻¹'
        }
    }
    
    # Handle plotting
    if output_settings.get('save_plots') and output_dir:
        plot_path = lambda name: os.path.join(output_dir, 'plots', 
                                            f'{name}_{timestamp}.{plot_format}')
    else:
        plot_path = lambda name: None

    plot_timelag_analysis(
        model, data,
        save_path=plot_path('timelag_analysis'),
        display=output_settings.get('display_plots', True)
    )
    
    if model.params.diffusivity and model.params.solubility:
        conc_profile, flux_data = model.solve_pde(
            D=model.params.diffusivity,
            C_eq=model.params.solubility,
            L=model.params.base.thickness,
            T=max(data['time']),
            dt=1.0,
            dx=0.01
        )
        
        plot_concentration_profile(
            conc_profile,
            save_path=plot_path('concentration_profile'),
            display=output_settings.get('display_plots', True)
        )
        
        plot_flux_over_time(
            flux_data,
            experimental_data=data,
            save_path=plot_path('flux_evolution'),
            display=output_settings.get('display_plots', True)
        )

    # Handle data saving
    if output_dir and output_settings.get('save_data'):
        if data_format == 'csv':
            data.to_csv(os.path.join(output_dir, 'data', f'raw_data_{timestamp}.csv'), 
                       index=False)
            pd.DataFrame(results_dict['results']).to_csv(
                os.path.join(output_dir, 'data', f'processed_data_{timestamp}.csv')
            )
        elif data_format == 'json':
            with open(os.path.join(output_dir, 'results', 
                                 f'model_results_{timestamp}.json'), 'w') as f:
                json.dump(results_dict, f, indent=4)

    return results_dict