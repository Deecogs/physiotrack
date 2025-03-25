import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

def analyze_rom(angles_file: Path) -> Dict[str, Any]:
    """
    Analyze range of motion from an angles file
    
    Args:
        angles_file: Path to the angles MOT file
        
    Returns:
        Dictionary with ROM analysis results
    """
    # Skip the header lines
    with open(angles_file, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('time'):
                header_rows = i
                break
    
    # Read the data
    angles_data = pd.read_csv(angles_file, sep='\t', skiprows=header_rows)
    
    # Calculate ROM for each joint
    rom_results = {}
    for col in angles_data.columns:
        if col == 'time':
            continue
        
        # Calculate min, max, and ROM
        min_val = float(angles_data[col].min())
        max_val = float(angles_data[col].max())
        rom = float(max_val - min_val)
        mean = float(angles_data[col].mean())
        std = float(angles_data[col].std())
        
        # Extract time series data for this angle
        time_series = [{
            "time": float(row["time"]),
            "value": float(row[col])
        } for _, row in angles_data.iterrows()]
        
        rom_results[col] = {
            "min": min_val,
            "max": max_val,
            "rom": rom,
            "mean": mean,
            "std": std,
            "time_series": time_series
        }
    
    return {
        "rom_analysis": rom_results,
        "summary": {
            "total_frames": len(angles_data),
            "duration": float(angles_data["time"].max()),
            "angles_measured": list(rom_results.keys())
        }
    }