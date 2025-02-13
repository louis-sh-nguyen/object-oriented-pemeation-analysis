import os
import pandas as pd
import matplotlib.pyplot as plt
from src.models.base_parameters import BaseParameters
import warnings
from src.utils.plotting import set_style
from src.models.single_pressure.variable_diffusivity_fvt import (
    FVTModel,
    FVTModelParameters,
    FVTTransportParams
)
from src.models.single_pressure.variable_diffusivity_fvt.plotting import (
    plot_diffusivity_profile,
    plot_diffusivity_location_profile,
    plot_norm_flux_over_time,
    plot_norm_flux_over_tau
)
from src.models.single_pressure.variable_diffusivity_fvt.workflow import (
    manual_workflow,
    data_fitting_workflow
)

def test_model_creation():
    """Test different ways to create FVTModel"""
    
    # Method 1: Simple parameter initialization
    model1 = FVTModel.from_parameters(
        pressure=50.0,          # Required base parameter
        temperature=25.0,       # Required base parameter
        thickness=0.1,          # Required transport parameter
        diameter=1.0,           # Required transport parameter
        flowrate=8.0,          # Optional transport parameter
        D1_prime=2.38,         # Optional FVT parameter
        DT_0=2.87e-7           # Optional FVT parameter
    )
    
    # Method 2: Explicit parameter objects
    base_params = BaseParameters(
        pressure=50.0,
        temperature=25.0
    )
    
    transport_params = FVTTransportParams(
        thickness=0.1,
        diameter=1.0,
        flowrate=8.0,
        D1_prime=2.38,
        DT_0=2.87e-7
    )
    
    model2 = FVTModel(FVTModelParameters(
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

def test_pde_solving():
    """Test PDE solving with FVT model"""
    model = FVTModel.from_parameters(
        pressure=50.0,
        temperature=25.0,
        thickness=0.1,
        diameter=1.0,
        flowrate=8.0,
        D1_prime=4.0,
        DT_0=1.0e-7
    )

    sim_params = {
        'T': 40e3,  # total time [s]
        'X': 1.0,     # normalized position
        'dx': 0.002,   # spatial step [adim]
    }
    
    # Solve PDE
    Dprime_df, flux_df = model.solve_pde(simulation_params=sim_params)
    
    print("\nPDE Solution Results:")
    print(f"Time points: {len(flux_df)}")
    print(f"Spatial points: {len(Dprime_df.columns)}")
    
    # Plot diffusivity-location profiles
    plot_diffusivity_location_profile(
        diffusivity_profile=Dprime_df,
        L=model.params.transport.thickness,
        T=sim_params['T'],
        # ax=ax2,
        display=True
    )
    
    # Plot normalized flux evolution
    plot_norm_flux_over_tau(
        flux_data=flux_df,
        # ax=ax3,
        display=True
    )
    
    # Plot flux evolution over time
    plot_norm_flux_over_time(
        flux_data=flux_df,
        # ax=ax3,
        display=True
    )
    
    # plt.tight_layout()
    # plt.show()

def test_manual_workflow():
    """Test the FVT workflow"""
    
    # Run workflow
    model, Dprime_df, flux_df, figures = manual_workflow(
        # data_path='data/single_pressure/RUN_H_25C-50bar.xlsx',
        pressure=50.0,
        temperature=25.0,
        thickness=0.1,
        diameter=1.0,
        flowrate=8.0,
        D1_prime=2.0,
        DT_0=2.87e-7,
        simulation_params={
            'T': 40e3,  # total time [s]
            'dx': 0.002,   # spatial step [adim]
            'X': 1.0      # normalized position
        },
        output_settings={
            'output_dir': 'outputs/manual_workflow',
            'display_plots': True,
            'save_plots': False,
            'save_data': False,
            'plot_format': 'png',
            'data_format': 'csv'
        }
    )
    
    # Print some results
    print("\nWorkflow Results:")
    print(f"Time points: {len(flux_df)}")
    print(f"Spatial points: {len(Dprime_df.columns)}")
    print(f"Max normalised flux: {flux_df['normalised_flux'].max():.4e}")
    print(f"Min normalised flux: {flux_df['normalised_flux'].min():.4e}")

def test_parameter_sensitivity():
    """Test combined effect of D1_prime and DT_0 on normalized flux vs tau curve"""
    
    # Base simulation parameters
    sim_params = {
        'T': 1e4,  # total time [s]
        'X': 1.0,      # normalized position
        'dx': 0.005   # spatial step [adim]
    }
    
    # Test different parameter combinations
    DT_0s = [1e-7, 1e-6]  # Range of DT_0 values
    D1_primes = [ 5.0, 50, ]  # Range of D1_prime values
    
    # Color map for D1_prime and line styles for DT_0
    colors = ['b', 'g', 'r']
    styles = [':', '--', '-']
    
    # Create figure
    set_style()
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Test all combinations
    for i, D1_prime in enumerate(D1_primes):
        for j, DT_0 in enumerate(DT_0s):
            model = FVTModel.from_parameters(
                pressure=50.0,
                temperature=25.0,
                thickness=0.1,
                diameter=1.0,
                flowrate=8.0,
                D1_prime=D1_prime,
                DT_0=DT_0
            )
            
            _, flux_df = model.solve_pde(simulation_params=sim_params)
            axs[0].plot(
                flux_df['tau'], flux_df['normalised_flux'], 
                color=colors[i], linestyle=styles[j],
                label=f'D1_prime={D1_prime:.1f}, DT_0={DT_0:.1e}'
            )
            axs[1].plot(
                flux_df['time'], flux_df['normalised_flux'],
                color=colors[i], linestyle=styles[j],
                label=f'D1_prime={D1_prime:.1f}, DT_0={DT_0:.1e}'
            )
    
    axs[0].set_xlabel(r'$\tau$ (Dimensionless Time)')
    axs[0].set_ylabel('Normalised Flux')
    axs[0].set_title(r'Normalised Flux vs $\tau$')
    
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Normalised Flux')
    axs[1].set_title('Normalised Flux vs Time')
    axs[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), fontsize='small')
    
    plt.tight_layout()
    plt.show()
    
def test_data_fitting_workflow_D1prime():
    """Test the FVT data fitting workflow"""
    # Turn off warning messages
    # warnings.filterwarnings('ignore')
    # Run workflow (optimization tracking is handled internally)
    model, fit_results, figures = data_fitting_workflow(
        data_path='data/single_pressure/RUN_H_25C-50bar.xlsx',
        pressure=50.0,
        temperature=25.0,
        thickness=0.1,
        diameter=1.0,
        flowrate=8.0,
        DT_0=2.8e-7,
        D1_prime=5.0,
        stabilisation_threshold=0.005,
        fitting_settings={
            'mode': 'D1',   # 'D1' or 'both'
            'initial_guess': 2.0,   # 5.0 or (5.0, 1e-7)
            'bounds': (1.001, 20),  # (1.001, 20) or ((1.001, 20), (1e-7, 1e-5))
            'n_starts': 1,  # 1, 2, 3 ,...
        },
        output_settings={
            'output_dir': 'outputs/fitting',
            'display_plots': True,
            'save_plots': False,
            'save_data': False,
            'plot_format': 'png',
            'data_format': 'csv'
        }
    )
    
    # Print fitting results
    print("\nFitting Results:")
    print(f"D1_prime: {fit_results['D1_prime']:.4e}") if 'D1_prime' in fit_results else None
    print(f'DT0: {fit_results["DT_0"]:.4e}') if 'DT_0' in fit_results else None
    print(f"RMSE: {fit_results['rmse']:.4e}")

def test_data_fitting_workflow_D1prime_DT0():
    """Test the FVT data fitting workflow"""
    # Turn off warning messages
    # warnings.filterwarnings('ignore')
    # Run workflow (optimization tracking is handled internally)
    model, fit_results, figures = data_fitting_workflow(
        data_path='data/single_pressure/S4R4.xlsx',
        pressure=50.0,
        temperature=25.0,
        thickness=0.1,
        diameter=1.0,
        flowrate=8.0,
        DT_0=2.8e-7,
        D1_prime=5.0,
        stabilisation_threshold=0.002,  # 0.005 for breakthrough curve, 0.002 for whole curve
        fitting_settings={
            'mode': 'both',   # 'D1' or 'both'
            'initial_guess': (5.0, 1e-7),   # 5.0 or (5.0, 1e-7)
            'bounds': ((1.001, 20), (1e-8, 1e-6)),  # (1.001, 20) or ((1.001, 20), (1e-7, 1e-5))
            'n_starts': 3,  # 1, 2, 3 ,...
        },
        output_settings={
            'output_dir': 'outputs/fitting',
            'display_plots': True,
            'save_plots': False,
            'save_data': False,
            'plot_format': 'png',
            'data_format': 'csv'
        }
    )
    
    # Print fitting results
    print("\nFitting Results:")
    print(f"D1_prime: {fit_results['D1_prime']:.4e}") if 'D1_prime' in fit_results else None
    print(f'DT0: {fit_results["DT_0"]:.4e}') if 'DT_0' in fit_results else None
    print(f"RMSE: {fit_results['rmse']:.4e}")

def fit_all_data(n=None):
    """Apply data_fitting_workflow to all xlsx files in data/single_pressure folder"""
    data_dir = 'data/single_pressure'
    
    # Create timestamp-based output directory
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_base_dir = f'outputs/fitting/{timestamp}'
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
        data_path = os.path.join(data_dir, file)
        
        try:
            # Extract temperature and pressure from filename
            # Assuming filename format: "RUN_X_##C-##bar.xlsx"
            try:
                parts = file.replace('.xlsx', '').split('_')[-1].split('-')
                temperature = float(parts[0].replace('C', ''))
                pressure = float(parts[1].replace('bar', ''))
            except:
                temperature = None
                pressure = None
            
            # Create file-specific output directory
            file_output_dir = os.path.join(output_base_dir, file[:-5])
            
            # Run workflow for this file
            model, fit_results, figures = data_fitting_workflow(
                data_path=data_path,
                pressure=pressure,
                temperature=temperature,
                thickness=0.1,
                diameter=1.0,
                flowrate=8.0,
                DT_0=2.8e-7,
                D1_prime=2.0,
                stabilisation_threshold=0.005,
                fitting_settings={
                    'mode': 'both',
                    'initial_guess': (2.0, 1e-7),
                    'bounds': ((1.001, 10), (1e-8, 1e-6)),
                    'n_starts': 3,
                },
                output_settings={
                    'output_dir': file_output_dir,
                    'display_plots': False,
                    'save_plots': True,
                    'save_data': True,
                    'plot_format': 'svg',
                    'data_format': 'csv'
                }
            )
            
            # Store essential results plus file name and conditions
            all_results[file] = {
                'file_name': file,
                'temperature': temperature,
                'pressure': pressure,
                'D1_prime': fit_results['D1_prime'],
                'DT_0': fit_results['DT_0'],
                'rmse': fit_results['rmse']
            }
            
            print(f"Successfully fitted {file}")
            print(f"T = {temperature}°C, P = {pressure} bar")
            print(f"D1_prime: {fit_results['D1_prime']:.4e}")
            print(f"DT_0: {fit_results['DT_0']:.4e}")
            print(f"RMSE: {fit_results['rmse']:.4e}")
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            all_results[file] = {
                'file_name': file,
                'error': str(e)
            }
    
    # Save overall results to CSV in timestamp directory
    results_df = pd.DataFrame.from_dict(all_results, orient='index')
    results_df.to_csv(os.path.join(output_base_dir, 'all_results.csv'), index=False)
    print(f"\nCompleted processing all files. Results saved to {output_base_dir}")

if __name__ == '__main__':
    # test_model_creation()
    # test_pde_solving()
    # test_manual_workflow()
    # test_parameter_sensitivity()
    # test_data_fitting_workflow_D1prime()
    test_data_fitting_workflow_D1prime_DT0()
    # fit_all_data()