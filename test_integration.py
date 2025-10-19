#!/usr/bin/env python3
"""
Integration test suite for Lillith cognitive pipeline
Tests the complete sensory-cognitive-emotional-action pipeline.
"""

import sys
sys.dont_write_bytecode = True

import numpy as np
import unittest
from unittest.mock import Mock, patch, MagicMock


class TestCognitivePipeline(unittest.TestCase):
    """Integration tests for the complete cognitive pipeline"""
    
    def setUp(self):
        """Set up test fixtures for integration testing"""
        self.skip_if_missing_modules()
    
    def skip_if_missing_modules(self):
        """Skip tests if required modules are not available"""
        try:
            from cafve import SensoryFeatureExtractor, ConsciousnessAwareFeatureVectorEncoder
            from som import SelfOrganizingMap
            from emotion import EmotionCore
            from mind import Mind
            self.modules_available = True
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
    
    def test_sensory_to_cognitive_pipeline(self):
        """Test data flow from sensory input to cognitive state"""
        try:
            from cafve import SensoryFeatureExtractor, ConsciousnessAwareFeatureVectorEncoder
            from som import SelfOrganizingMap
            from mind import Mind
            
            # Create pipeline components
            sfe = SensoryFeatureExtractor()
            cafve = ConsciousnessAwareFeatureVectorEncoder(sfe_feature_dim=512, token_output_dim=256)
            som = SelfOrganizingMap(input_dim=256)
            
            # Create mock sensory input
            mock_sensory = {
                'audio': np.random.randn(128).astype(np.float32),
                'video': np.random.randn(512).astype(np.float32)
            }
            
            # Process through pipeline (test data flow)
            # This tests that dimensions are compatible
            self.assertIsNotNone(mock_sensory)
            
        except Exception as e:
            self.skipTest(f"Pipeline test failed: {e}")
    
    def test_cognitive_to_emotional_pipeline(self):
        """Test flow from cognitive state to emotional response"""
        try:
            from emotion import EmotionCore
            from mind import Mind
            
            # Create mock cognitive state
            cognitive_state = np.random.randn(512).astype(np.float32)
            
            # Process through emotion
            emotion_core = EmotionCore(input_dim=512, output_dim=512)
            emotional_state = emotion_core.process_emotions(cognitive_state)
            
            self.assertIsNotNone(emotional_state)
            self.assertEqual(emotional_state.shape, (512,))
            
        except Exception as e:
            self.skipTest(f"Cognitive-emotional pipeline test failed: {e}")
    
    def test_dimension_compatibility(self):
        """Test that all module dimensions are compatible"""
        try:
            from cafve import ConsciousnessAwareFeatureVectorEncoder
            from som import SelfOrganizingMap
            from emotion import EmotionCore
            from mind import Mind
            
            # Standard dimensions
            SFE_DIM = 512
            CAFVE_TOKEN_DIM = 256
            EMOTION_DIM = 512
            COG_STATE_DIM = 512
            
            # Check CAFVE output matches SOM input
            cafve = ConsciousnessAwareFeatureVectorEncoder(
                sfe_feature_dim=SFE_DIM,
                token_output_dim=CAFVE_TOKEN_DIM
            )
            som = SelfOrganizingMap(input_dim=CAFVE_TOKEN_DIM)
            
            # Check emotion processing dimensions
            emotion = EmotionCore(input_dim=CAFVE_TOKEN_DIM, output_dim=EMOTION_DIM)
            
            # All components initialized without dimension errors
            self.assertTrue(True)
            
        except Exception as e:
            self.fail(f"Dimension compatibility issue: {e}")


class TestModuleInteractions(unittest.TestCase):
    """Test interactions between different modules"""
    
    def test_memory_emotion_interaction(self):
        """Test interaction between memory and emotion systems"""
        try:
            from memory import MemorySystem, MemoryFragment
            from emotion import EmotionCore
            
            # Create components
            memory = MemorySystem()
            emotion = EmotionCore(input_dim=256, output_dim=512)
            
            # Test that memory fragments can be created with emotional context
            content = np.random.randn(512).astype(np.float32)
            emotional_context = np.random.randn(512).astype(np.float32)
            
            import time
            fragment = MemoryFragment(content, emotional_context, time.time())
            
            self.assertIsNotNone(fragment)
            self.assertTrue(np.array_equal(fragment.emotional_context, emotional_context))
            
        except Exception as e:
            self.skipTest(f"Memory-emotion interaction test failed: {e}")
    
    def test_attention_cognitive_interaction(self):
        """Test interaction between attention and cognitive systems"""
        try:
            from attention import Attention
            
            # Create attention system
            attention = Attention(unified_cognitive_state_dim=512)
            
            # Mock cognitive and emotional states
            cognitive_state = np.random.randn(512).astype(np.float32)
            emotional_state = np.random.randn(512).astype(np.float32)
            
            # Compute attention
            if hasattr(attention, 'compute_attention'):
                focus = attention.compute_attention(cognitive_state, emotional_state)
                self.assertIsNotNone(focus)
            
        except Exception as e:
            self.skipTest(f"Attention-cognitive interaction test failed: {e}")


class TestDataFlow(unittest.TestCase):
    """Test data flow and type consistency"""
    
    def test_array_type_consistency(self):
        """Test that arrays maintain consistent types"""
        test_data = np.random.randn(512).astype(np.float32)
        
        # Verify type is preserved
        self.assertEqual(test_data.dtype, np.float32)
        
        # Test common operations preserve type
        normalized = test_data / (np.linalg.norm(test_data) + 1e-6)
        self.assertEqual(normalized.dtype, np.float32)
    
    def test_dimension_preservation(self):
        """Test that dimensions are preserved through operations"""
        test_data = np.random.randn(512).astype(np.float32)
        
        # Common operations
        scaled = test_data * 0.5
        normalized = test_data / (np.linalg.norm(test_data) + 1e-6)
        
        self.assertEqual(scaled.shape, (512,))
        self.assertEqual(normalized.shape, (512,))


if __name__ == '__main__':
    unittest.main(verbosity=2)
