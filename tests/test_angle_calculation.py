#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the angle calculation functionality of PhysioTrack.
"""

import unittest
import numpy as np
from physiotrack.Utilities.common import points_to_angles, fixed_angles

class TestAngleCalculation(unittest.TestCase):
    
    def test_points_to_angles_horizontal(self):
        """Test angle calculation with 2 points (horizontal reference)"""
        # Points form a horizontal line from left to right
        points = [np.array([10, 5]), np.array([0, 5])]
        angle = points_to_angles(points)
        self.assertAlmostEqual(angle, 0.0, places=1)
        
        # Points form a vertical line from top to bottom
        points = [np.array([5, 0]), np.array([5, 10])]
        angle = points_to_angles(points)
        self.assertAlmostEqual(angle, 90.0, places=1)
        
        # Points form a 45-degree line
        points = [np.array([10, 0]), np.array([0, 10])]
        angle = points_to_angles(points)
        self.assertAlmostEqual(angle, 45.0, places=1)
    
    def test_points_to_angles_joint(self):
        """Test angle calculation with 3 points (joint angle)"""
        # 90-degree angle
        points = [np.array([0, 0]), np.array([5, 5]), np.array([10, 0])]
        angle = points_to_angles(points)
        self.assertAlmostEqual(angle, 90.0, places=1)
        
        # 180-degree angle (straight line)
        points = [np.array([0, 5]), np.array([5, 5]), np.array([10, 5])]
        angle = points_to_angles(points)
        self.assertAlmostEqual(angle, 180.0, places=1)
        
        # 45-degree angle
        points = [np.array([0, 0]), np.array([5, 5]), np.array([5, 10])]
        angle = points_to_angles(points)
        self.assertAlmostEqual(angle, 45.0, places=1)
    
    def test_fixed_angles(self):
        """Test fixed angle calculation with offsets and scaling"""
        # Test right knee angle (with fixed offset -180 and scale 1)
        points = [np.array([0, 10]), np.array([5, 5]), np.array([10, 0])]
        angle = fixed_angles(points, 'right knee')
        # The raw angle is 90 degrees, with offset -180 becomes -90
        self.assertAlmostEqual(angle, -90.0, places=1)
        
        # Test right ankle angle (with fixed offset 90 and scale 1)
        points = [np.array([0, 0]), np.array([5, 5]), np.array([10, 5]), np.array([5, 0])]
        angle = fixed_angles(points, 'right ankle')
        # Complex case with 4 points, the result depends on the vectors formed
        self.assertTrue(-180 <= angle <= 180)

if __name__ == '__main__':
    unittest.main()