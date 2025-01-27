import os
import pandas as pd
from src.models.single_pressure.constant_diffusivity.workflow import time_lag_analysis_workflow
from src.models.single_pressure.constant_diffusivity.plotting import plot_concentration_profile, plot_flux_over_time

def main():
    # Define file path
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'single_pressure')
    file_path = os.path.join(data_dir, 'RUN_H_25C-50bar.xlsx')
    
    # Define model parameters
    thickness = 0.1  # cm
    diameter = 1.0  # cm
    flow_rate = 8.0  # ml/min
    pressure = 50.0  # bar
    temperature = 25+273  # K
    
    # Run time-lag analysis workflow
    output_dict = time_lag_analysis_workflow(
        file_path, thickness, diameter, flow_rate, pressure, temperature,
        output_settings={
            'output_dir': 'output',
            'display_plots': True,
            'save_plots': False,
            'save_data': False,
            'plot_format': 'png',  # Choose one: 'png' or 'svg'
            'data_format': 'csv'   # Choose one: 'csv' or 'json'
        }
    )
    results = output_dict['results']
    
    # Print results
    print("Analysis Results:")
    print(f"Diffusivity: {results['diffusivity']:.4e} cm^2/s")
    print(f"Permeability: {results['permeability']:.4e} barrer")
    print(f"Solubility: {results['solubility']:.4e} cm^3(STP)/(cm^3⋅bar)")
    

if __name__ == '__main__':
    main()