import os
from pathlib import Path

def safe_long_path(path):
    """Convert path to long path format if needed for Windows"""
    if os.name == 'nt' and len(str(path)) > 250:  # Windows only, leave margin
        if isinstance(path, str):
            path = Path(path)
        return f'\\\\?\\{path.resolve()}'
    return str(path)