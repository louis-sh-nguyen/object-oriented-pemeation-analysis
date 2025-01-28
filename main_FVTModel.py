import os
import pandas as pd
from src.models.single_pressure.variable_diffusivity_fvt import FVTModel

def test_model_manually():
    # Create model instance with manual parameters
    model = FVTModel.from_parameters(
        thickness=0.1,          # [cm]
        diameter=1.0,           # [cm]
        flowrate=8.0,         # [cm³(STP) s⁻¹]
        pressure=50.0,          # [bar]
        temperature=25.0,       # [°C]
        D1_prime=2.38,
        D2_prime=1.00,
        D0_T=2.87e-7
    )
    
    # Solve PDE
    T = 1000  # total time [s]
    dt = 1.0  # time step [s]
    dx = 0.001  # spatial step [cm]
    
    Dprime_df, flux_df = model.solve_pde(
        D1_prime=model.params.transport.D1_prime,
        D2_prime=model.params.transport.D2_prime,
        D0_T=model.params.transport.D0_T,
        L=model.params.base.thickness,
        T=T,
        dt=dt,
        dx=dx
    )

if __name__ == '__main__':
    test_model_manually()