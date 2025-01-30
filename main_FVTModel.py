import os
import pandas as pd
import matplotlib.pyplot as plt
from src.models.base_parameters import BaseParameters
from src.models.single_pressure.variable_diffusivity_fvt import (
    FVTModel,
    FVTModelParameters,
    FVTTransportParams
)
from src.models.single_pressure.variable_diffusivity_fvt.plotting import (
    plot_diffusivity_profile,
    plot_diffusivity_location_profile,
    plot_flux_over_time,
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
        D1_prime=2.38,
        DT_0=2.87e-7
    )
    
    # Set simulation parameters
    sim_params = {
        'T': 100000,  # total time [s]
        'dt': 1.0,    # time step [s]
        'dx': 0.01,   # spatial step [adim]
        'X': 1.0      # normalized position
    }
    
    # Solve PDE
    Dprime_df, flux_df = model.solve_pde(simulation_params=sim_params)
    
    print("\nPDE Solution Results:")
    print(f"Time points: {len(flux_df)}")
    print(f"Spatial points: {len(Dprime_df.columns)}")
    print(f"Max flux: {flux_df['flux'].max():.4e}")
    print(f"Min flux: {flux_df['flux'].min():.4e}")
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))
    
    # Plot diffusivity profile evolution
    plot_diffusivity_profile(
        diffusivity_profile=Dprime_df,
        ax=ax1,
        display=False
    )
    
    # Plot diffusivity-location profiles
    plot_diffusivity_location_profile(
        diffusivity_profile=Dprime_df,
        L=model.params.transport.thickness,
        T=sim_params['T'],
        ax=ax2,
        display=False
    )
    
    # Plot flux evolution
    plot_flux_over_time(
        flux_data=flux_df,
        ax=ax3,
        display=False
    )
    
    # Plot normalized flux evolution
    plot_norm_flux_over_tau(
        flux_data=flux_df,
        ax=ax4,
        display=False
    )
    
    plt.tight_layout()
    plt.show()

def test_manual_workflow():
    """Test the FVT workflow"""
    
    # Run workflow
    model, Dprime_df, flux_df, figures = manual_workflow(
        pressure=50.0,
        temperature=25.0,
        thickness=0.1,
        diameter=1.0,
        flowrate=8.0,
        D1_prime=2.38,
        DT_0=2.87e-7,
        simulation_params={
            'T': 100000,  # total time [s]
            'dt': 1.0,    # time step [s]
            'dx': 0.01,   # spatial step [adim]
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
    print(f"Max flux: {flux_df['flux'].max():.4e}")
    print(f"Min flux: {flux_df['flux'].min():.4e}")

def test_data_fitting_workflow():
    """Test the FVT data fitting workflow"""
    # Run workflow (optimization tracking is handled internally)
    model, fit_results, figures = data_fitting_workflow(
        data_path='data/single_pressure/RUN_H_25C-50bar.xlsx',
        pressure=50.0,
        temperature=25.0,
        thickness=0.1,
        diameter=1.0,
        flowrate=8.0,
        initial_guess={
            'D1_prime': 2.38,
            'DT_0': 2.87e-7
        },
        output_settings={
            'output_dir': 'outputs/fitting',
            'display_plots': True,
            'save_plots': True,
            'save_data': True,
            'plot_format': 'png',
            'data_format': 'csv'
        }
    )
    
    # Print fitting results
    print("\nFitting Results:")
    print(f"D1_prime: {fit_results['D1_prime']:.4e}")
    print(f"DT_0: {fit_results['DT_0']:.4e}")
    print(f"RMSE: {fit_results['rmse']:.4e}")

if __name__ == '__main__':
    # test_model_creation()
    # test_pde_solving()
    # test_manual_workflow()
    test_data_fitting_workflow()