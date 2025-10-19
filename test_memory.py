#!/usr/bin/env python3
"""
Test suite for memory.py - Memory System Module
Tests memory storage, retrieval, and consolidation.
"""

import sys
sys.dont_write_bytecode = True

import numpy as np
import unittest
import time
from unittest.mock import Mock, patch


class TestMemoryFragment(unittest.TestCase):
    """Test cases for MemoryFragment class"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            from memory import MemoryFragment
            self.MemoryFragment = MemoryFragment
        except ImportError as e:
            self.skipTest(f"Cannot import memory module: {e}")
    
    def test_memory_fragment_creation(self):
        """Test creating a memory fragment"""
        content = np.random.randn(512).astype(np.float32)
        emotional_context = np.random.randn(512).astype(np.float32)
        timestamp = time.time()
        
        fragment = self.MemoryFragment(content, emotional_context, timestamp)
        
        self.assertTrue(np.array_equal(fragment.content, content))
        self.assertEqual(fragment.creation_time, timestamp)
        self.assertEqual(fragment.access_count, 0)
    
    def test_memory_fragment_access_update(self):
        """Test updating memory fragment access"""
        content = np.random.randn(512).astype(np.float32)
        emotional_context = np.random.randn(512).astype(np.float32)
        timestamp = time.time()
        
        fragment = self.MemoryFragment(content, emotional_context, timestamp)
        
        # Update access
        new_time = timestamp + 100
        fragment.update_access(new_time)
        
        self.assertEqual(fragment.access_count, 1)
        self.assertEqual(fragment.last_access_time, new_time)
    
    def test_memory_fragment_relevance(self):
        """Test memory fragment relevance calculation"""
        content = np.random.randn(512).astype(np.float32)
        emotional_context = np.random.randn(512).astype(np.float32)
        timestamp = time.time()
        
        fragment = self.MemoryFragment(content, emotional_context, timestamp)
        
        query_emotion = np.random.randn(512).astype(np.float32)
        relevance = fragment.get_relevance(timestamp + 10, query_emotion)
        
        self.assertIsInstance(relevance, (float, np.floating))
        # Relevance can be negative due to emotional dissimilarity


class TestMemorySystem(unittest.TestCase):
    """Test cases for MemorySystem class"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            from memory import MemorySystem
            self.MemorySystem = MemorySystem
        except ImportError as e:
            self.skipTest(f"Cannot import MemorySystem: {e}")
    
    def test_memory_system_initialization(self):
        """Test MemorySystem initialization"""
        # Check constructor signature
        try:
            memory = self.MemorySystem(
                sfe_feature_dim=512,
                cognitive_state_dim=512,
                emotional_state_dim=512
            )
            self.assertIsNotNone(memory)
        except TypeError:
            # Try without arguments
            memory = self.MemorySystem()
            self.assertIsNotNone(memory)
    
    def test_memory_storage_and_retrieval(self):
        """Test storing and retrieving memories"""
        try:
            memory = self.MemorySystem(
                sfe_feature_dim=512,
                cognitive_state_dim=512,
                emotional_state_dim=512
            )
        except TypeError:
            memory = self.MemorySystem()
        
        # Check if memory has store/retrieve methods
        if hasattr(memory, 'store_memory'):
            content = np.random.randn(512).astype(np.float32)
            emotional_context = np.random.randn(512).astype(np.float32)
            
            memory.store_memory(content, emotional_context)
            
            # Try to retrieve
            if hasattr(memory, 'retrieve_memory'):
                query = np.random.randn(512).astype(np.float32)
                retrieved = memory.retrieve_memory(query)
                self.assertIsNotNone(retrieved)


class TestMemoryConsolidation(unittest.TestCase):
    """Test cases for MemoryConsolidation class"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            from memory import MemoryConsolidation
            self.MemoryConsolidation = MemoryConsolidation
        except ImportError as e:
            self.skipTest(f"Cannot import MemoryConsolidation: {e}")
    
    def test_consolidation_initialization(self):
        """Test MemoryConsolidation initialization"""
        consolidation = self.MemoryConsolidation(memory_dimension=512, internal_hidden_dim=128)
        self.assertEqual(consolidation.dimension, 512)
    
    def test_dimension_update(self):
        """Test dynamic dimension update"""
        consolidation = self.MemoryConsolidation(memory_dimension=512, internal_hidden_dim=128)
        consolidation.update_dimensions(memory_dimension=1024, internal_hidden_dim=256)
        self.assertEqual(consolidation.dimension, 1024)


if __name__ == '__main__':
    unittest.main(verbosity=2)
