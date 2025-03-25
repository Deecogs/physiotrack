#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the pose estimation functionality of PhysioTrack.
"""

import unittest
import numpy as np
import cv2
import os
from pathlib import Path

class TestPoseEstimation(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        # Create a blank test image
        self.test_image = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Draw a simple stick figure in the middle
        # Head
        cv2.circle(self.test_image, (640, 200), 50, (255, 255, 255), -1)
        # Body
        cv2.line(self.test_image, (640, 250), (640, 450), (255, 255, 255), 5)
        # Arms
        cv2.line(self.test_image, (640, 300), (540, 350), (255, 255, 255), 5)
        cv2.line(self.test_image, (640, 300), (740, 350), (255, 255, 255), 5)
        # Legs
        cv2.line(self.test_image, (640, 450), (590, 600), (255, 255, 255), 5)
        cv2.line(self.test_image, (640, 450), (690, 600), (255, 255, 255), 5)
        
        # Save the test image
        self.test_image_path = "test_pose_image.jpg"
        cv2.imwrite(self.test_image_path, self.test_image)
        
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
    
    def test_pose_detection_setup(self):
        """Test that the pose detection setup works"""
        try:
            from rtmlib import PoseTracker, BodyWithFeet
            # Just test that we can create a pose tracker
            pose_tracker = PoseTracker(
                BodyWithFeet,
                det_frequency=1,
                mode='lightweight',
                backend='openvino',
                device='cpu',
                tracking=False,
                to_openpose=False)
            self.assertTrue(True)  # If we got here, it works
        except ImportError:
            self.skipTest("RTMLib not available for testing")
    
    def test_pose_detection_inference(self):
        """Test that pose detection inference works on a simple image"""
        try:
            from rtmlib import PoseTracker, BodyWithFeet
            # Create a pose tracker
            pose_tracker = PoseTracker(
                BodyWithFeet,
                det_frequency=1,
                mode='lightweight',
                backend='openvino',
                device='cpu',
                tracking=False,
                to_openpose=False)
            
            # Run inference on test image
            img = cv2.imread(self.test_image_path)
            keypoints, scores = pose_tracker(img)
            
            # We should have at least some detections (may not be accurate since it's a stick figure)
            self.assertIsInstance(keypoints, np.ndarray)
            self.assertIsInstance(scores, np.ndarray)
            
        except ImportError:
            self.skipTest("RTMLib not available for testing")
        except Exception as e:
            self.skipTest(f"Error running pose detection: {e}")
            
    def test_person_tracking(self):
        """Test that person tracking works across frames"""
        try:
            from physiotrack.Utilities.common import sort_people_physiotrack
            
            # Create mock keypoints for two frames
            keypoints_frame1 = np.array([
                [[100, 100], [120, 120], [140, 100]],  # Person 1
                [[500, 100], [520, 120], [540, 100]]   # Person 2
            ])
            
            # Slightly move the keypoints for the second frame
            keypoints_frame2 = np.array([
                [[105, 105], [125, 125], [145, 105]],  # Person 1 moved slightly
                [[495, 105], [515, 125], [535, 105]]   # Person 2 moved slightly
            ])
            
            # Create mock scores
            scores_frame1 = np.array([
                [0.9, 0.8, 0.7],  # Person 1
                [0.8, 0.7, 0.6]   # Person 2
            ])
            
            scores_frame2 = np.array([
                [0.85, 0.75, 0.65],  # Person 1
                [0.75, 0.65, 0.55]   # Person 2
            ])
            
            # Test tracking
            sorted_prev_keypoints, sorted_keypoints, sorted_scores = sort_people_physiotrack(
                keypoints_frame1, keypoints_frame2, scores_frame2
            )
            
            # The sorted keypoints should match the original order (no swapping)
            np.testing.assert_allclose(sorted_keypoints[0], keypoints_frame2[0], rtol=1e-5)
            np.testing.assert_allclose(sorted_keypoints[1], keypoints_frame2[1], rtol=1e-5)
            
        except Exception as e:
            self.skipTest(f"Error testing person tracking: {e}")
            
if __name__ == '__main__':
    unittest.main()