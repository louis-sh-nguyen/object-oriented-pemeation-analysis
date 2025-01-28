import os
import pandas as pd
import json
from src.models.single_pressure.constant_diffusivity.workflow import time_lag_analysis_workflow
from src.models.single_pressure.constant_diffusivity.plotting import plot_concentration_profile, plot_flux_over_time
from src.models.parameters import BaseParameters
from src.models.single_pressure.constant_diffusivity.model import TimelagModel


def test_model_manually():
    # Create model instance with manual parameters
    model = TimelagModel.from_manual_parameters(
        thickness=0.1,          # cm
        diameter=1.0,           # cm
        flowrate=8.0,         # cm³(STP) s⁻¹
        pressure=50.0,          # bar
        temperature=25.0,       # °C
        diffusivity=1e-7,      # cm² s⁻¹
        equilibrium_concentration=0.1  # cm³(STP) cm⁻³
    )
    
    # Solve PDE
    T = 1000  # total time [s]
    dt = 1.0  # time step [s]
    dx = 0.001  # spatial step [cm]
    
    conc_profile, flux = model.solve_pde(
        D=model.params.diffusivity,
        C_eq=model.params.equilibrium_concentration,
        L=model.params.base.thickness,
        T=T,
        dt=dt,
        dx=dx
    )
    
    # Plot results
    plot_concentration_profile(conc_profile)
    plot_flux_over_time(flux)

def test_model_fitting():
    # Load experimental data
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'single_pressure')
    file_path = os.path.join(data_dir, 'RUN_H_25C-50bar.xlsx')
    data = pd.read_excel(file_path)
    
    # Create model instance
    model = TimelagModel.from_manual_parameters(
        thickness=0.1,          # cm
        diameter=1.0,           # cm
        flowrate=8.0,         # cm³(STP) s⁻¹
        pressure=50.0,          # bar
        temperature=25.0,       # °C
    )
    
    # Fit model to data
    model.fit_to_data(data)
    
    # Print fitted parameters
    print("\nFitted Parameters:")
    print(f"Diffusivity: {model.params.diffusivity:.4e} cm²/s")
    print(f"Permeability: {model.params.permeability:.4e} cm³(STP) cm⁻¹ s⁻¹ bar⁻¹")
    print(f"Solubility: {model.params.solubility_coefficient:.4e} cm³(STP)/(cm³⋅bar)")
    print(f"Equilibrium concentration: {model.params.equilibrium_concentration:.4e} cm³(STP)/cm³")

def test_workflow():
    # Define file path
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'single_pressure')
    file_path = os.path.join(data_dir, 'RUN_H_25C-50bar.xlsx')
    
    # Define model parameters
    thickness = 0.1  # cm
    diameter = 1.0  # cm
    flow_rate = 8.0  # ml/min
    pressure = 50.0  # bar
    temperature = 25  # °C
    
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
    print(f"Permeability: {results['permeability']:.4e} cm³(STP) cm⁻¹ s⁻¹ bar⁻¹")
    print(f"Solubility: {results['equilibrium_concentration']:.4e} cm^3(STP)/(cm^3⋅bar)")
    
if __name__ == '__main__':
    # test_model_manually()
    # test_model_fitting()
    test_workflow()