import os
import pandas as pd
from src.models.base_parameters import BaseParameters
from src.models.single_pressure.constant_diffusivity import *

def test_parameter_methods():
    """Test different ways to create TimelagModel"""
    
    # Method 1: From individual parameters
    model1 = TimelagModel.from_parameters(
        thickness=0.1,
        diameter=2.0,
        flowrate=8.0,
        pressure=50.0,
        temperature=25.0,
        diffusivity=1e-6,
        equilibrium_concentration=1e-2
    )
    
    # Method 2: From parameter objects
    base_params = BaseParameters(
        thickness=0.1,
        diameter=2.0,
        flowrate=8.0,
        pressure=50.0,
        temperature=25.0
    )
    
    transport_params = TimelagTransportParams(
        diffusivity=1e-6,
        equilibrium_concentration=1e-2
    )
    
    model2 = TimelagModel(TimelagModelParameters(
        base=base_params,
        transport=transport_params
    ))
    
    # Print parameters
    print("\nModel 1 Parameters:")
    print(f"Base Parameters: {model1.params.base.__dict__}")
    print(f"Transport Parameters: {model1.params.transport.__dict__}")
    
    print("\nModel 2 Parameters:")
    print(f"Base Parameters: {model2.params.base.__dict__}")
    print(f"Transport Parameters: {model2.params.transport.__dict__}")

def test_model_fitting():
    """Test model fitting with experimental data"""
    # Load experimental data
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'single_pressure')
    file_path = os.path.join(data_dir, 'RUN_H_25C-50bar.xlsx')
    data = pd.read_excel(file_path)
    
    # Create and fit model
    model = TimelagModel.from_parameters(
        thickness=0.1,
        diameter=1.0,
        flowrate=8.0,
        pressure=50.0,
        temperature=25.0
    )
    
    processed_data = model.fit_to_data(data)
    
    # Print results
    print("\nFitted Parameters:")
    print(f"Diffusivity: {model.results['diffusivity']:.4e} cm²/s")
    print(f"Permeability: {model.results['permeability']:.4e} cm³(STP) cm⁻¹ s⁻¹ bar⁻¹")
    print(f"Solubility: {model.results['solubility_coefficient']:.4e} cm³(STP)/(cm³⋅bar)")

def test_workflow():
    """Test complete workflow"""
    results = time_lag_analysis_workflow(
        file_path='data/single_pressure/RUN_H_25C-50bar.xlsx',
        thickness=0.1,
        diameter=1.0,
        flowrate=8.0,
        pressure=50.0,
        temperature=25.0,
        output_settings={
            'output_dir': 'output/test_workflow',
            'display_plots': True,
            'save_plots': False,
            'save_data': False
        }
    )
    
    print("\nWorkflow Results:")
    print(f"Diffusivity: {results['results']['diffusivity']:.4e} cm²/s")
    print(f"Permeability: {results['results']['permeability']:.4e} cm³(STP) cm⁻¹ s⁻¹ bar⁻¹")
    print(f"Time Lag: {results['results']['time_lag']:.2f} s")

if __name__ == '__main__':
    # test_parameter_methods()
    # test_model_fitting()
    test_workflow()