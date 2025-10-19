#!/usr/bin/env python3
"""
Test suite for emotion.py - Emotional Processing Module
Tests emotional state management and processing.
"""

import sys
sys.dont_write_bytecode = True

import numpy as np
import unittest
from unittest.mock import Mock, patch


class TestEmotionalState(unittest.TestCase):
    """Test cases for EmotionalState class"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            from emotion import EmotionalState
            self.EmotionalState = EmotionalState
        except ImportError as e:
            self.skipTest(f"Cannot import emotion module: {e}")
    
    def test_emotional_state_initialization(self):
        """Test EmotionalState initialization"""
        state = self.EmotionalState(base_dimension=512)
        self.assertEqual(state.dimension, 512)
        self.assertEqual(state.valence.shape, (512,))
        self.assertEqual(state.arousal.shape, (512,))
        self.assertEqual(state.dominance.shape, (512,))
        self.assertEqual(state.intensity.shape, (512,))
    
    def test_emotional_state_update(self):
        """Test updating emotional state"""
        state = self.EmotionalState(base_dimension=512)
        
        valence = np.random.randn(512).astype(np.float32)
        arousal = np.random.randn(512).astype(np.float32)
        dominance = np.random.randn(512).astype(np.float32)
        intensity = np.random.randn(512).astype(np.float32)
        
        state.update(valence, arousal, dominance, intensity)
        
        self.assertTrue(np.array_equal(state.valence, valence))
        self.assertTrue(np.array_equal(state.arousal, arousal))
        self.assertEqual(len(state.temporal_context), 1)
    
    def test_unified_state_generation(self):
        """Test unified emotional state generation"""
        state = self.EmotionalState(base_dimension=512)
        
        valence = np.ones(512, dtype=np.float32) * 0.5
        arousal = np.ones(512, dtype=np.float32) * 0.3
        dominance = np.ones(512, dtype=np.float32) * 0.7
        intensity = np.ones(512, dtype=np.float32) * 0.9
        
        state.update(valence, arousal, dominance, intensity)
        unified = state.get_unified_state()
        
        self.assertEqual(unified.shape, (512,))
    
    def test_temporal_context_limit(self):
        """Test that temporal context is limited to 100 entries"""
        state = self.EmotionalState(base_dimension=512)
        
        for i in range(150):
            valence = np.random.randn(512).astype(np.float32)
            arousal = np.random.randn(512).astype(np.float32)
            dominance = np.random.randn(512).astype(np.float32)
            intensity = np.random.randn(512).astype(np.float32)
            state.update(valence, arousal, dominance, intensity)
        
        self.assertEqual(len(state.temporal_context), 100)


class TestEmotionCore(unittest.TestCase):
    """Test cases for EmotionCore class"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            from emotion import EmotionCore
            self.EmotionCore = EmotionCore
        except ImportError as e:
            self.skipTest(f"Cannot import EmotionCore: {e}")
    
    def test_emotion_core_initialization(self):
        """Test EmotionCore initialization"""
        # EmotionCore takes input_dim and output_dim
        core = self.EmotionCore(input_dim=256, output_dim=512)
        self.assertIsNotNone(core)
    
    def test_emotion_processing(self):
        """Test emotion processing from cognitive state"""
        core = self.EmotionCore(input_dim=256, output_dim=512)
        
        # Create mock cognitive state
        cognitive_state = np.random.randn(256).astype(np.float32)
        
        # Process emotions
        emotional_output = core.process_emotions(cognitive_state)
        
        # Check output dimension
        self.assertEqual(emotional_output.shape, (512,))


class TestEmotionalModulation(unittest.TestCase):
    """Test cases for EmotionalModulation class"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            from emotion import EmotionalModulation
            self.EmotionalModulation = EmotionalModulation
        except ImportError as e:
            self.skipTest(f"Cannot import EmotionalModulation: {e}")
    
    def test_modulation_initialization(self):
        """Test EmotionalModulation initialization"""
        modulation = self.EmotionalModulation(emotion_dim=512, cognitive_dim=256)
        self.assertEqual(modulation.emotion_dimension, 512)
        self.assertEqual(modulation.cognitive_dimension, 256)


if __name__ == '__main__':
    unittest.main(verbosity=2)
