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
    
    # Plot results
    # fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(16, 4))
    
    # Plot diffusivity profile evolution
    # plot_diffusivity_profile(
    #     diffusivity_profile=Dprime_df,
    #     ax=ax1,
    #     display=False
    # )
    
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
    axs[0].set_title('Normalised Flux vs Tau')
    
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Normalised Flux')
    axs[1].set_title('Normalised Flux vs Time')
    axs[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), fontsize='x-small')
    
    plt.tight_layout()
    plt.show()
    
def test_data_fitting_workflow():
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
        DT_0=2.87e-7,
        fitting_settings={
            'mode': 'D1',   # 'd1' or 'both'
            'initial_guess': 5.0,   # 5.0 or (5.0, 1e-7)
            'bounds': (1.001, 20),  # (1.001, 20) or ((1.001, 20), (1e-7, 1e-5))
            'n_starts': 1
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
    print(f"D1_prime: {fit_results['D1_prime']:.4e}")
    print(f"RMSE: {fit_results['rmse']:.4e}")

if __name__ == '__main__':
    # test_model_creation()
    # test_pde_solving()
    test_manual_workflow()
    # test_parameter_sensitivity()
    # test_data_fitting_workflow()