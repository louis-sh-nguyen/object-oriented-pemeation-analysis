import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
import os

def set_style() -> None:
    """Configure default plotting style"""
    plt.style.use('classic')
    plt.rcParams.update({
        'font.size': 10,
        'figure.figsize': (5, 4),
        'lines.linewidth': 2,
        'axes.labelsize': 'smaller'
    })

