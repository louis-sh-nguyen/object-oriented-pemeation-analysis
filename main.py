import os
import pandas as pd
import json
from src.models.single_pressure.constant_diffusivity.workflow import time_lag_analysis_workflow
from src.models.single_pressure.constant_diffusivity.plotting import plot_concentration_profile, plot_flux_over_time
from src.models.base_parameters import BaseParameters, ModelParameters, TransportParams
from src.models.single_pressure.constant_diffusivity.model import TimelagModel


def test_model_manually():
    # Create model instance with manual parameters
    model = TimelagModel.from_parameters(
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
        D=model.params.transport.diffusivity,
        C_eq=model.params.transport.equilibrium_concentration,
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
    model = TimelagModel.from_parameters(
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
    print(f"Diffusivity: {model.params.transport.diffusivity:.4e} cm²/s")
    print(f"Permeability: {model.params.transport.permeability:.4e} cm³(STP) cm⁻¹ s⁻¹ bar⁻¹")
    print(f"Solubility: {model.params.transport.solubility_coefficient:.4e} cm³(STP)/(cm³⋅bar)")
    print(f"Equilibrium concentration: {model.params.transport.equilibrium_concentration:.4e} cm³(STP)/cm³")

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

def test_parameter_methods():
    """Example usage of different parameter methods"""
    
    # Method 1: from_parameter_objects (Advanced/Internal Use)
    base = BaseParameters(
        thickness=0.1,
        diameter=2.0,
        flowrate=8.0,
        pressure=50.0,
        temperature=25.0
    )
    
    transport = TransportParams(
        diffusivity=1e-6,
        equilibrium_concentration=1e-2
    )
    
    model1 = TimelagModel.from_parameter_objects(base, transport)
    
    # Method 2: from_parameters (User-Friendly Interface)
    model2 = TimelagModel.from_parameters(
        thickness=0.1,
        diameter=2.0,
        flowrate=8.0,
        pressure=50.0,
        temperature=25.0,
        diffusivity=1e-6,
        equilibrium_concentration=1e-2
    )

    # Print parameters for both models
    print("\nModel 1 Parameters:")
    print(f"Base Parameters: {model1.params.base.__dict__}")
    print(f"Transport Parameters: {model1.params.transport.__dict__}")
    
    print("\nModel 2 Parameters:")
    print(f"Base Parameters: {model2.params.base.__dict__}")
    print(f"Transport Parameters: {model2.params.transport.__dict__}")

if __name__ == '__main__':
    # test_model_manually()
    # test_model_fitting()
    # test_workflow()
    test_parameter_methods()