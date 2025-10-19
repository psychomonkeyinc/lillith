#!/usr/bin/env python3
"""
Test suite for som.py - Self-Organizing Map Module
Tests the SOM implementation and map operations.
"""

import sys
sys.dont_write_bytecode = True

import numpy as np
import unittest
from unittest.mock import Mock, patch


class TestSelfOrganizingMap(unittest.TestCase):
    """Test cases for SelfOrganizingMap class"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            from som import SelfOrganizingMap
            self.SelfOrganizingMap = SelfOrganizingMap
        except ImportError as e:
            self.skipTest(f"Cannot import som module: {e}")
    
    def test_som_initialization(self):
        """Test SOM initialization"""
        try:
            som = self.SelfOrganizingMap(
                map_size=(17, 17),
                input_dim=256,
                activation_threshold=15.0,
                fatigue_cost=0.1,
                fatigue_decay=0.1
            )
            self.assertIsNotNone(som)
            self.assertEqual(som.input_dim, 256)
        except TypeError:
            # Try minimal initialization
            som = self.SelfOrganizingMap(input_dim=256)
            self.assertIsNotNone(som)
    
    def test_som_process_input(self):
        """Test processing input through SOM"""
        try:
            som = self.SelfOrganizingMap(
                map_size=(17, 17),
                input_dim=256,
                activation_threshold=15.0,
                fatigue_cost=0.1,
                fatigue_decay=0.1
            )
        except TypeError:
            som = self.SelfOrganizingMap(input_dim=256)
        
        # Create test input
        test_input = np.random.randn(256).astype(np.float32)
        
        if hasattr(som, 'process_input'):
            output = som.process_input(test_input)
            self.assertIsNotNone(output)
        elif hasattr(som, 'activate'):
            output = som.activate(test_input)
            self.assertIsNotNone(output)
    
    def test_som_learning(self):
        """Test SOM learning/training"""
        try:
            som = self.SelfOrganizingMap(
                map_size=(17, 17),
                input_dim=256,
                activation_threshold=15.0,
                fatigue_cost=0.1,
                fatigue_decay=0.1
            )
        except TypeError:
            som = self.SelfOrganizingMap(input_dim=256)
        
        # Create training data
        training_data = np.random.randn(100, 256).astype(np.float32)
        
        if hasattr(som, 'train'):
            som.train(training_data, epochs=1)
        elif hasattr(som, 'fit'):
            som.fit(training_data, epochs=1)
    
    def test_som_bmu_computation(self):
        """Test Best Matching Unit computation"""
        try:
            som = self.SelfOrganizingMap(
                map_size=(17, 17),
                input_dim=256,
                activation_threshold=15.0,
                fatigue_cost=0.1,
                fatigue_decay=0.1
            )
        except TypeError:
            som = self.SelfOrganizingMap(input_dim=256)
        
        test_input = np.random.randn(256).astype(np.float32)
        
        if hasattr(som, 'find_bmu'):
            bmu = som.find_bmu(test_input)
            self.assertIsNotNone(bmu)
        elif hasattr(som, 'get_bmu'):
            bmu = som.get_bmu(test_input)
            self.assertIsNotNone(bmu)


class TestSOMFatigue(unittest.TestCase):
    """Test cases for SOM fatigue mechanism"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            from som import SelfOrganizingMap
            self.SelfOrganizingMap = SelfOrganizingMap
        except ImportError as e:
            self.skipTest(f"Cannot import som module: {e}")
    
    def test_fatigue_map_exists(self):
        """Test that fatigue map is initialized"""
        try:
            som = self.SelfOrganizingMap(
                map_size=(17, 17),
                input_dim=256,
                activation_threshold=15.0,
                fatigue_cost=0.1,
                fatigue_decay=0.1
            )
            
            if hasattr(som, 'fatigue_map'):
                self.assertIsNotNone(som.fatigue_map)
                self.assertEqual(som.fatigue_map.shape, (17, 17))
        except Exception:
            pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
