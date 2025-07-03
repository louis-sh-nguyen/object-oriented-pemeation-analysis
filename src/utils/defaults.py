
"""Default values and mappings for experimental parameters"""

# Default values for common parameters
DEFAULTS = {
    "diameter": 1.0,        # cm
    "flowrate": 8.0,        # cm³/min
    "thickness": 0.1,       # mm
}

# Sample-specific thickness mappings
THICKNESS_DICT = {
    'RUN_H_25C-50bar': 0.1, 
    'RUN_H_25C-100bar_7': 0.1, 
    'RUN_H_25C-100bar_8': 0.1, 
    'RUN_H_25C-100bar_9': 0.1, 
    'RUN_H_25C-200bar_2': 0.1,
    'RUN_H_50C-50bar': 0.1, 
    'RUN_H_50C-100bar_2': 0.1, 
    'RUN_H_50C-200bar': 0.1, 
    'RUN_H_75C-50bar': 0.1, 
    'RUN_H_75C-100bar': 0.1,
    'S3R1': 0.1, 
    'S3R2': 0.1, 
    'S3R3': 0.1, 
    'S3R4': 0.1,
    'S4R3': 0.025, 
    'S4R4': 0.025, 
    'S4R5': 0.025, 
    'S4R6': 0.025,
} # [cm]

# Sample-specific flowrate mappings
FLOWRATE_DICT = {
    'RUN_H_25C-50bar': 8.0, 
    'RUN_H_25C-100bar_7': 8.0, 
    'RUN_H_25C-100bar_8': 8.0, 
    'RUN_H_25C-100bar_9': 8.0, 
    'RUN_H_25C-200bar_2': 8.0,
    'RUN_H_50C-50bar': 8.0, 
    'RUN_H_50C-100bar_2': 8.0, 
    'RUN_H_50C-200bar': 8.0, 
    'RUN_H_75C-50bar': 8.0, 
    'RUN_H_75C-100bar': 8.0,
    'S3R1': 4.17, 
    'S3R2': 4.046, 
    'S3R3': 4.027, 
    'S3R4': 4.0454, 
    'S4R3': 9.83, 
    'S4R4': 9.84, 
    'S4R5': 9.92, 
    'S4R6': 10,
}  # [ml min^-1]

# Default fitting parameters for Variable FVT model
FVT_FITTING_DEFAULTS = {
    "D1_prime": {
        "lower_bound": 1.001,
        "upper_bound": 20.0,
        "initial": 2.0
    },
    "DT0": {
        "lower_bound": 1e-11,
        "upper_bound": 1e-6,
        "initial": 1e-7
    },
    "n_starts": 1
}

# Default fitting parameters for Constant Diffusivity model
CONSTANT_D_FITTING_DEFAULTS = {
    "D0": {
        "lower_bound": 1e-13,
        "upper_bound": 1e-11,
        "initial": 1e-12
    },
    "n_starts": 1
}

# Temperature mapping for samples
TEMPERATURE_DICT = {
    'RUN_H_25C-50bar': 25.0, 
    'RUN_H_25C-100bar_7': 25.0, 
    'RUN_H_25C-100bar_8': 25.0, 
    'RUN_H_25C-100bar_9': 25.0, 
    'RUN_H_25C-200bar_2': 25.0,
    'RUN_H_50C-50bar': 50.0, 
    'RUN_H_50C-100bar_2': 50.0, 
    'RUN_H_50C-200bar': 50.0, 
    'RUN_H_75C-50bar': 75.0, 
    'RUN_H_75C-100bar': 75.0,
    
    'S3R1': 115, 
    'S3R2': 115, 
    'S3R3': 115, 
    'S3R4': 115, 
    'S4R3': 25, 
    'S4R4': 50, 
    'S4R5': 75, 
    'S4R6': 50,
}

# Pressure mapping for samples
PRESSURE_DICT = {
    'RUN_H_25C-50bar': 50.0, 
    'RUN_H_25C-100bar_7': 100.0, 
    'RUN_H_25C-100bar_8': 100.0, 
    'RUN_H_25C-100bar_9': 100.0, 
    'RUN_H_25C-200bar_2': 200.0,
    'RUN_H_50C-50bar': 50.0, 
    'RUN_H_50C-100bar_2': 100.0, 
    'RUN_H_50C-200bar': 200.0, 
    'RUN_H_75C-50bar': 50.0, 
    'RUN_H_75C-100bar': 100.0,
    
    'S3R1': 100, 
    'S3R2': 200, 
    'S3R3': 300, 
    'S3R4': 100, 
    'S4R3': 25, 
    'S4R4': 50, 
    'S4R5': 50, 
    'S4R6': 50,
} # [bar]