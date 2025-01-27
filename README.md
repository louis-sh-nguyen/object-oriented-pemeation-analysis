# Time-Lag Analysis for Gas Permeation

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

## License

[Add license information]

## Contact

Louis Nguyen
sn621@ic.ac.uk