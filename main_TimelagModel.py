import os
import pandas as pd
from src.models.single_pressure.constant_diffusivity import (
    TimelagModel,
    TimelagModelParameters,
    TimelagTransportParams,
    time_lag_analysis_workflow,
    plot_timelag_analysis
)
from src.models.base_parameters import BaseParameters

def test_model_creation():
    """Test different ways to create TimelagModel"""
    
    # Method 1: Simple parameter initialization
    model1 = TimelagModel.from_parameters(
        pressure=50.0,        # Required base parameter
        temperature=25.0,     # Required base parameter
        thickness=0.1,        # Required transport parameter
        diameter=2.0,         # Required transport parameter
        diffusivity=1e-6,     # Optional transport parameter
        equilibrium_concentration=1e-2  # Optional transport parameter
    )
    
    # Method 2: Explicit parameter objects
    base_params = BaseParameters(
        pressure=50.0,
        temperature=25.0
    )
    
    transport_params = TimelagTransportParams(
        thickness=0.1,
        diameter=2.0,
        diffusivity=1e-6,
        equilibrium_concentration=1e-2
    )
    
    model2 = TimelagModel(TimelagModelParameters(
        base=base_params,
        transport=transport_params
    ))
    
    # Print parameters to verify
    print("\nModel 1 Parameters:")
    print(f"Base Parameters: {model1.params.base.__dict__}")
    print(f"Transport Parameters: {model1.params.transport.__dict__}")
    
    print("\nModel 2 Parameters:")
    print(f"Base Parameters: {model2.params.base.__dict__}")
    print(f"Transport Parameters: {model2.params.transport.__dict__}")

def test_data_fitting():
    """Test model fitting with experimental data"""
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'single_pressure')
    file_path = os.path.join(data_dir, 'RUN_H_25C-50bar.xlsx')
    data = pd.read_excel(file_path)
    
    # Create model with minimum required parameters
    model = TimelagModel.from_parameters(
        pressure=50.0,
        temperature=25.0,
        thickness=0.1,
        diameter=1.0,
        flowrate=8.0,
    )
    
    # Fit to data
    processed_data = model.fit_to_data(data)
    
    # Print results
    print("\nFitted Parameters:")
    for param, value in model.results.items():
        print(f"{param}: {value:.4e}")

def test_full_workflow():
    """Test complete analysis workflow"""
    results = time_lag_analysis_workflow(
        file_path='data/single_pressure/RUN_H_25C-50bar.xlsx',
        pressure=50.0,
        temperature=25.0,
        thickness=0.1,
        diameter=1.0,
        flowrate=8.0,
        output_settings={
            'output_dir': 'output/test_workflow',
            'display_plots': True,
            'save_plots': False,
            'save_data': False,
        }
    )
    
    print("\nWorkflow Results:")
    for param, value in results['results'].items():
        if isinstance(value, float):
            print(f"{param}: {value:.4e}")

if __name__ == '__main__':
    # test_model_creation()
    # test_data_fitting()
    test_full_workflow()