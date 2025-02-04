# Gas Permeation Analysis

A Python package for analyzing gas permeation data using the time-lag method.

## Overview

This package provides tools for:
- Time-lag analysis of gas permeation data
- Calculation of diffusivity, permeability, and solubility coefficients
- Visualization of permeation data and model fits
- Solution of the permeation PDE for concentration profiles

## Installation

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd Code_new
pip install -r requirements.txt
```

## Usage

Basic usage example:

```python
from src.models.single_pressure.constant_diffusivity.workflow import time_lag_analysis_workflow

# Run analysis
results = time_lag_analysis_workflow(
    file_path='path/to/data.xlsx',
    thickness=0.1,  # cm
    diameter=1.0,   # cm
    flow_rate=8.0,  # ml/min
    pressure=50.0,  # bar
    temperature=25.0,  # °C
    output_settings={
        'output_dir': 'output',
        'display_plots': True,
        'save_plots': True,
        'save_data': True,
        'plot_format': 'png',  # 'png' or 'svg'
        'data_format': 'csv'   # 'csv' or 'json'
    }
)
```

## Input Data Format

The input Excel file should contain:
- Time data (seconds)
- Flux measurements (cm³(STP) cm⁻² s⁻¹)

## Output Options

### Plots
- Time-lag analysis
- Concentration profiles
- Flux evolution over time

### Data Formats
- CSV: Raw data and processed results
- JSON: Model parameters and results

### Plot Formats
- PNG: Standard bitmap format
- SVG: Vector graphics format

## Output Directory Structure

```
output/
├── data/
│   ├── raw_data_*.csv
│   └── processed_data_*.csv
├── results/
│   └── model_results_*.json
└── plots/
    ├── timelag_analysis_*.png
    ├── concentration_profile_*.png
    └── flux_evolution_*.png
```

## Units

- Thickness: cm
- Diameter: cm
- Flow rate: cm³(STP) s⁻¹
- Pressure: bar
- Temperature: °C
- Diffusivity: cm² s⁻¹
- Permeability: cm³(STP) cm⁻¹ s⁻¹ bar⁻¹
- Solubility: cm³(STP) cm⁻³ bar⁻¹

## Parameter Structure and Model Workflow

### Parameter Classes
```
parameters/
├── base_params/
│   ├── thickness
│   ├── diameter
│   ├── flowrate
│   ├── pressure
│   └── temperature
├── transport_params/
│   ├── diffusivity
│   ├── permeability
│   └── solubility
└── model_params/
    ├── base_params
    ├── transport_params
```
### Data Classes Hierarchy
1. BaseParameters
   - Stores experimental conditions (thickness, diameter, flowrate, pressure, temperature)
   - Basic validation for physical parameters
   - Used independently for data processing

2. TransportParams
   - Contains transport properties (diffusivity, permeability, solubility)
   - Optional parameters that can be fitted or specified manually
   - Independent validation for transport parameters

3. ModelParameters
   - Combines BaseParameters and TransportParams
   - Complete parameter set for model initialization
   - Cascading validation of all parameters

## Usage Pattern
1. Parameters (Dataclass)
   - Data storage
   - Validation
   - Type safety

2. Model Creation (Classmethod)
   - from_parameters()
   - from_data()

3. Model Interface (Abstract)
   - fit_to_data()
   - calculate_diffusivity()
   - calculate_permeability()

## Algorithm Description
### Model: VariableDiffusivityFVT

### Overview
The solver implements an implicit Newton method with adaptive time stepping and a fallback strategy to robustly solve the PDE in the FVT model. The implementation consists of two major components:

1. **Newton Method for Time-Stepping:**  
   Iteratively updates the solution at each time step until the relative residual meets the specified tolerance.

2. **Adaptive Time Stepping with Fallback:**  
   Dynamically adjusts the time step based on convergence. If the Newton method fails to converge at a given dt, the time step is reduced. When dt becomes too small, the solver activates a fallback mode by fixing dt and increasing damping to secure convergence.

### Newton Method Implementation
- **Residual Calculation:**  
  For each interior grid point, the residual is computed using the discretized PDE:
  
  \[
  R_i = \frac{D_i^\text{new} - D_i^\text{old}}{dt} - K \, D_i^\text{new} \, \Delta_x D
  \]
  
  where \(\Delta_x D\) is approximated by a central finite difference.

- **Convergence Check:**  
  The algorithm checks for convergence by comparing the relative \(L^2\) norm of the residual to a tolerance:
  
  \[
  \frac{\|R\|_2}{\|D\|_2} < \text{rel_tol}
  \]
  
  If the condition is met, the iteration stops.

- **Relaxed Newton Update:**  
  A diagonal approximation to the Jacobian is used to correct the solution:
  
  \[
  D_i^\text{new} \leftarrow D_i^\text{new} - \text{relax} \cdot \frac{R_i}{J_{ii}}
  \]
  
  The relaxation factor helps improve stability. Boundary conditions are enforced after each update.

### Adaptive Time Stepping with Fallback
- **Adaptive Strategy:**  
  The solver begins with an initial time step (`dt_init`). If the Newton update is successful, dt is gradually increased (using a ramp-up factor) until it reaches a target value (`dt_target`).

- **Handling Convergence Failures:**  
  If the Newton iterations fail to converge:
  - The time step is halved and the update is retried.
  - When the time step falls below a minimum threshold (`dt_min`), the solver does not stop. Instead, it switches to a fallback mode:
    - **Fallback Mode:**  
      dt is fixed to `dt_min`, and the damping is increased (i.e., the relaxation factor is reduced) to help achieve convergence under challenging conditions.

- **Recording the Solution:**  
  The complete time-history is stored as two DataFrames:
  - `Dprime_df`: diffusivity profile over space and time.
  - `flux_df`: normalized flux data computed at each time step.

This combination of Newton's method with adaptive time stepping and a fallback strategy ensures robust convergence even when the system exhibits stiff behavior or challenging convergence properties.

## License

[Add license information]

## Contact

Louis Nguyen
sn621@ic.ac.uk