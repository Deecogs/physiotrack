#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the PhysioTrack library with a webcam input.
This is a simple demo that shows how to use PhysioTrack with a webcam.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run PhysioTrack with webcam input."""
    print("Starting PhysioTrack with webcam input...")
    
    # Set up command to run PhysioTrack with webcam
    cmd = [
        "physiotrack",
        "--video_input", "webcam",
        "--show_realtime_results", "True",
        "--save_vid", "True",
        "--save_img", "False",
        "--show_graphs", "False",
        "--det_frequency", "4",
        "--mode", "performance"
    ]
    # mode = 'performance' # 'lightweight', 'balanced', 'performance',
    print(f"Command: {' '.join(cmd)}")
    print("Press 'q' to quit.")
    
    # Run PhysioTrack
    try:
        result = subprocess.run(cmd, check=True)
        print("PhysioTrack completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running PhysioTrack: {e}")
    except KeyboardInterrupt:
        print("PhysioTrack stopped by user.")
    
if __name__ == "__main__":
    main()