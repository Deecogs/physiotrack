# PhysioTrack

**PhysioTrack is a computer vision library for physiotherapy assessment that automatically computes 2D joint positions, joint angles, and segment angles from a video or webcam.**

## Features

- **Pose Estimation**: Detect human keypoints from videos or webcam feeds
- **Joint and Segment Angle Calculation**: Compute range of motion and movement kinematics
- **Real-time Feedback**: Visualize joints, angles, and movement in real-time
- **Multi-person Tracking**: Follow multiple subjects throughout a video
- **Data Analysis**: Interpolate missing data, apply filters, and generate reports
- **Physiotherapy Assessment**: Analyze range of motion and movement quality

## Installation

### Quick Install

````bash
pip install physiotrack
For GPU acceleration (optional but recommended):
bashCopypip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install onnxruntime-gpu
Usage
Basic Usage
bashCopy# Run on demo video
physiotrack

# Run on custom video
physiotrack --video_input path_to_video.mp4

# Run with webcam
physiotrack --video_input webcam

# Analyze specific time range
physiotrack --video_input path_to_video.mp4 --time_range 1.5 5.0
Configuration Options
PhysioTrack can be configured using command-line arguments or a TOML configuration file:
bashCopy# Run with config file
physiotrack --config path_to_config.toml

# Run with specific parameters
physiotrack --video_input video.mp4 --multiperson false --mode lightweight --det_frequency 4
Visualization and Output

Real-time Display: View pose estimation and angle calculations in real-time
Video Output: Save processed video with overlaid poses and angles
Data Files: Export joint coordinates (TRC) and angle measurements (MOT)
Analysis Plots: Generate plots showing range of motion and movement patterns

Documentation
For detailed documentation, please visit https://github.com/yourusername/physiotrack.
Examples
Range of Motion Assessment
pythonCopyfrom physiotrack import PhysioTrack

# Configure assessment parameters
config = {
    'project': {
        'video_input': ['patient_assessment.mp4'],
        'px_to_m_person_height': 1.75,
        'visible_side': ['right']
    },
    'angles': {
        'joint_angles': ['Right knee', 'Right hip', 'Right shoulder'],
        'segment_angles': ['Right thigh', 'Right shank', 'Trunk']
    }
}

# Run assessment
PhysioTrack.process(config)
Real-time Exercise Guidance
bashCopyphysiotrack --video_input webcam --mode lightweight --det_frequency 4 --show_realtime_results true --display_angle_values_on body
License
PhysioTrack is released under the BSD 3-Clause License.
Acknowledgments
PhysioTrack is built on the foundation of modern pose estimation techniques and biomechanical analysis methods.
Copy
## Test Files

You'll need to get a sample video file to use as demo.mp4. You can use any video of a person performing various movements, especially focusing on movements relevant to physical therapy assessments.

## Next Steps

Now that you have the complete code for Phase 1 of PhysioTrack, you can get started implementing it. Here's a sequence of steps to follow:

1. **Set up the directory structure**:
   ```bash
   mkdir -p physiotrack/physiotrack/Demo physiotrack/physiotrack/Utilities physiotrack/tests

Create the files:

Copy each of the code sections to their appropriate files
Make sure to get a demo video file for testing


Install the library:
bashCopycd physiotrack
pip install -e .

Test the basic functionality:
bashCopy# Run the CLI command
physiotrack --help

# Try the demo
physiotrack

Run unit tests:
bashCopypython -m unittest discover -s tests
````
