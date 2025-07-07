from datetime import datetime
import os
import pandas as pd
from src.models.single_pressure.constant_diffusivity import (
    TimelagModel,
    TimelagModelParameters,
    TimelagTransportParams,
    data_fitting_workflow,
    manual_workflow
)
from src.models.base_parameters import BaseParameters
from src.utils.dir_paths import safe_long_path
from src.utils.defaults import TEMPERATURE_DICT, PRESSURE_DICT, DIAMETER_DICT, THICKNESS_DICT, FLOWRATE_DICT

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

def test_manual_workflow():
    """Test the FVT workflow"""
    
    # Run workflow
    model, conc_profile, flux_data, figures = manual_workflow(
        pressure=50.0,
        temperature=25.0,
        thickness=0.1,
        diameter=1.0,
        flowrate=8.0,
        diffusivity=7.00e-09,
        equilibrium_concentration=50,
        simulation_params={
            'dt': 5.0,  # [s]
            'T': 700e3,  # total time [s]
            'dx': 0.001,   # spatial step [adim]
            'X': 1.0,      # normalized position
        },
        output_settings={
            'output_dir': 'outputs/manual_workflow',
            'display_plots': True,
            'save_plots': False,
            'save_data': True,
            'plot_format': 'png',
            'data_format': 'csv'
        }
    )
    
    # Print some results
    print("\nWorkflow Results:")
    print(f"Time points: {len(flux_data)}")
    print(f"Spatial points: {len(conc_profile.columns)}")
    print(f"Max flux: {flux_data['flux'].max():.4e}")
    print(f"Min flux: {flux_data['flux'].min():.4e}")

def test_full_workflow():
    """Test complete analysis workflow"""
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'single_pressure')
    file_path = os.path.join(data_dir, 'RUN_H_25C-50bar.xlsx')
    
    results = data_fitting_workflow(
        file_path=file_path,
        pressure=50.0,
        temperature=25.0,
        thickness=0.1,
        diameter=1.0,
        flowrate=8.0,
        output_settings={
            'output_dir': os.path.join(os.path.dirname(__file__), 'output', 'test_workflow'),
            'display_plots': True,
            'save_plots': False,
            'save_data': False,
        }
    )
    
    print("\nWorkflow Results:")
    for param, value in results['results'].items():
        if isinstance(value, float):
            print(f"{param}: {value:.4e}")

def fit_all_data(n=None):
    """Fit all data in the single pressure directory"""
    data_dir = 'data/single_pressure'
    
    # Create timestamp-based output directory
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_base_dir = os.path.join(base_dir, f'outputs/fitting/{timestamp}')
    os.makedirs(output_base_dir, exist_ok=True)
    
    # List all xlsx files in the data directory
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    
    # Limit number of files to process
    if n is not None:
        data_files = data_files[:n]
    
    # Store results for all files
    all_results = {}
    
    for file in data_files:
        print(f"\nProcessing {file}...")
        data_path = safe_long_path(os.path.join(data_dir, file))
        exp_name = file.replace('.xlsx', '')
        try:
            # Get temperature and pressure
            temperature = TEMPERATURE_DICT.get(exp_name, None)
            pressure = PRESSURE_DICT.get(exp_name, None)
            thickness = THICKNESS_DICT.get(exp_name, 0.1)
            diameter = DIAMETER_DICT.get(exp_name, 1.0)
            flowrate = FLOWRATE_DICT.get(exp_name, 8.0)
            
            # Create file-specific output directory
            file_output_dir = os.path.join(output_base_dir, file[:-5])

            # Run workflow
            model_with_results, results_dict, figures, conc_profile, flux_data, processed_exp_data = data_fitting_workflow(
                file_path=data_path,
                pressure=pressure,
                temperature=temperature,
                thickness=thickness,
                diameter=diameter,
                flowrate=flowrate,
                stabilisation_threshold=0.001,  # 0.005 for breakthrough curve, 0.002 for whole curve
                output_settings={
                    'output_dir': file_output_dir,
                    'display_plots': False,
                    'save_plots': True,
                    'save_data': True,
                    'plot_format': 'pdf',
                    'data_format': 'csv',
                }
            )
            
            # Store results
            all_results[file] = {
                'file_name': file,
                'temperature': model_with_results.params.base.temperature,
                'pressure': model_with_results.params.base.pressure,
                'thickness': model_with_results.params.transport.thickness,
                'diameter': model_with_results.params.transport.diameter,
                'flowrate': model_with_results.params.transport.flowrate,
                'timelag': model_with_results.results.get('time_lag', None),
                'diffusivity': model_with_results.results.get('diffusivity', None),
                'permeability': model_with_results.results.get('permeability', None),                
                'equilibrium_concentration': model_with_results.results.get('equilibrium_concentration', None),
                'rmse': model_with_results.results.get('rmse', None),
                'r2': model_with_results.results.get('r2', None),
            }
            
            # Print results for each file
            print(f"Successfully fitted {file}")
            for key, value in all_results[file].items():
                print(f"{key}: {value}")            
                print('')
        
        except Exception as e:
            print(f"Error processing {file}: {e}")
            all_results[file] = {
                'file_name': file,
                'error': str(e)
            }
    
    # Save overall results to CSV in timestamp directory
    results_path = os.path.join(output_base_dir, 'timelag_all_results.csv')
    results_path = safe_long_path(results_path)
    results_df = pd.DataFrame.from_dict(all_results, orient='index')
    results_df.to_csv(results_path, index=False)
    print(f"\nCompleted processing all files. Results saved to {output_base_dir}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory set to: {os.getcwd()}")
    
    # test_model_creation()
    # test_data_fitting()
    # test_manual_workflow()
    # test_full_workflow()
    fit_all_data()