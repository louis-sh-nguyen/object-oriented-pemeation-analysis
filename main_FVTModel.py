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
    plot_flux_over_time
)
from src.models.single_pressure.variable_diffusivity_fvt.workflow import fvt_workflow

def test_model_creation():
    """Test different ways to create FVTModel"""
    
    # Method 1: Simple parameter initialization
    model1 = FVTModel.from_parameters(
        pressure=50.0,          # Required base parameter
        temperature=25.0,       # Required base parameter
        thickness=0.1,          # Required transport parameter
        diameter=1.0,           # Required transport parameter
        D1_prime=2.38,         # Optional FVT parameter
        D2_prime=1.00,         # Optional FVT parameter
        D0_T=2.87e-7           # Optional FVT parameter
    )
    
    # Method 2: Explicit parameter objects
    base_params = BaseParameters(
        pressure=50.0,
        temperature=25.0
    )
    
    transport_params = FVTTransportParams(
        thickness=0.1,
        diameter=1.0,
        D1_prime=2.38,
        D2_prime=1.00,
        D0_T=2.87e-7
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
        D1_prime=2.38,
        D2_prime=1.00,
        D0_T=2.87e-7
    )
    
    # Solve PDE
    T = 100000  # total time [s]
    dt = 1.0  # time step [s]
    dx = 0.01  # spatial step [adim]
    
    Dprime_df, flux_df = model.solve_pde(
        D1_prime=model.params.transport.D1_prime,
        D2_prime=model.params.transport.D2_prime,
        D0_T=model.params.transport.D0_T,
        T=T,
        X=1.0,
        L=model.params.transport.thickness,
        dt=dt,
        dx=dx
    )
    
    print("\nPDE Solution Results:")
    print(f"Time points: {len(flux_df)}")
    print(f"Spatial points: {len(Dprime_df.columns)}")
    print(f"Max flux: {flux_df['flux'].max():.4e}")
    print(f"Min flux: {flux_df['flux'].min():.4e}")
    
    # Plot concentration profile evolution
    plot_diffusivity_profile(
        diffusivity_profile=Dprime_df,
        display=True
    )
    
    # Plot concentration-location profiles
    plot_diffusivity_location_profile(
        diffusivity_profile=Dprime_df,
        L=model.params.transport.thickness,
        T=T,
        display=True
    )
    
    # Plot flux evolution using plot_flux_over_time
    plot_flux_over_time(
        flux_data=flux_df,
        display=True
    )

def test_workflow():
    """Test the FVT workflow"""
    
    # Example simulation parameters
    sim_params = {
        'T': 100000,  # total time [s]
        'dt': 1.0,    # time step [s]
        'dx': 0.01,   # spatial step [adim]
        'X': 1.0      # normalized position
    }
    
    # Run workflow
    model, Dprime_df, flux_df, figures = fvt_workflow(
        pressure=50.0,
        temperature=25.0,
        thickness=0.1,
        diameter=1.0,
        D1_prime=2.38,
        D2_prime=1.00,
        D0_T=2.87e-7,
        simulation_params=sim_params,
        output_dir='outputs'
    )
    
    # Print some results
    print("\nWorkflow Results:")
    print(f"Time points: {len(flux_df)}")
    print(f"Spatial points: {len(Dprime_df.columns)}")
    print(f"Max flux: {flux_df['flux'].max():.4e}")
    print(f"Min flux: {flux_df['flux'].min():.4e}")

if __name__ == '__main__':
    # test_model_creation()
    # test_pde_solving()
    test_workflow()