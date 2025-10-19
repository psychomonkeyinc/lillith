# nvme_memory.py - NVME Storage Integration for Persistent Memory Management

import numpy as np
import os
import mmap
import pickle
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import time
import psutil
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NVMEMemoryManager:
    """
    NVME-optimized memory manager for persistent storage and memory-mapped arrays.
    Provides high-performance persistent memory with automatic cleanup and optimization.
    """

    def __init__(self, base_path: str = "./nvme_memory", max_memory_gb: float = 8.0):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Memory limits and monitoring
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self.current_memory_usage = 0

        # Memory-mapped file registry
        self.mapped_files: Dict[str, mmap.mmap] = {}
        self.array_shapes: Dict[str, Tuple] = {}
        self.array_dtypes: Dict[str, np.dtype] = {}

        # Performance tracking
        self.access_counts: Dict[str, int] = {}
        self.last_access: Dict[str, float] = {}

        # Automatic cleanup settings
        self.cleanup_threshold = 0.8  # Clean up when 80% memory used
        self.max_age_days = 30  # Remove files older than 30 days

        logger.info(f"NVME Memory Manager initialized at {base_path}")

    def _get_memory_usage(self) -> float:
        """Get current memory usage as fraction of max"""
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / self.max_memory_bytes

    def _cleanup_old_files(self):
        """Remove old memory files to free up space"""
        current_time = time.time()
        cutoff_time = current_time - (self.max_age_days * 24 * 3600)

        for file_path in self.base_path.glob("*.npy"):
            if file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    logger.info(f"Cleaned up old file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {file_path}: {e}")

    def create_memory_mapped_array(self, name: str, shape: Tuple, dtype: np.dtype = np.float16) -> np.ndarray:
        """
        Create a memory-mapped array for persistent storage.
        Automatically handles file creation and memory mapping.
        """
        file_path = self.base_path / f"{name}.npy"

        # Check if we need cleanup
        if self._get_memory_usage() > self.cleanup_threshold:
            self._cleanup_old_files()
            gc.collect()

        try:
            # Create or load existing array
            if file_path.exists():
                # Load existing array
                array = np.load(file_path, mmap_mode='r+')
                logger.info(f"Loaded existing memory-mapped array: {name} {array.shape}")
            else:
                # Create new array
                array = np.zeros(shape, dtype=dtype)
                np.save(file_path, array)
                array = np.load(file_path, mmap_mode='r+')
                logger.info(f"Created new memory-mapped array: {name} {shape}")

            # Register the array
            self.array_shapes[name] = shape
            self.array_dtypes[name] = dtype
            self.access_counts[name] = 0
            self.last_access[name] = time.time()

            return array

        except Exception as e:
            logger.error(f"Failed to create memory-mapped array {name}: {e}")
            # Fallback to regular numpy array
            return np.zeros(shape, dtype=dtype)

    def save_array(self, name: str, array: np.ndarray):
        """Save array to NVME storage with compression optimization"""
        file_path = self.base_path / f"{name}.npy"

        try:
            # Ensure array is in memory-mapped mode if possible
            if not hasattr(array, 'filename') or array.filename is None:
                # Save as memory-mapped
                np.save(file_path, array)
                # Reload as memory-mapped for future access
                mapped_array = np.load(file_path, mmap_mode='r+')
                return mapped_array
            else:
                # Already memory-mapped, just flush
                array.flush()
                return array

        except Exception as e:
            logger.error(f"Failed to save array {name}: {e}")
            return array

    def load_array(self, name: str) -> Optional[np.ndarray]:
        """Load array from NVME storage"""
        file_path = self.base_path / f"{name}.npy"

        if not file_path.exists():
            logger.warning(f"Array {name} not found in NVME storage")
            return None

        try:
            array = np.load(file_path, mmap_mode='r+')
            self.access_counts[name] = self.access_counts.get(name, 0) + 1
            self.last_access[name] = time.time()
            return array
        except Exception as e:
            logger.error(f"Failed to load array {name}: {e}")
            return None

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        total_files = len(list(self.base_path.glob("*.npy")))
        total_size = sum(f.stat().st_size for f in self.base_path.glob("*.npy"))

        return {
            'total_files': total_files,
            'total_size_mb': total_size / (1024**2),
            'memory_usage_percent': self._get_memory_usage() * 100,
            'mapped_arrays': len(self.array_shapes),
            'most_accessed': max(self.access_counts.items(), key=lambda x: x[1], default=(None, 0))
        }

class OptimizedMemorySystem:
    """
    Optimized memory system with NVME integration and vectorized operations.
    Replaces the original memory system with performance improvements.
    """

    def __init__(self, nvme_manager: NVMEMemoryManager, dimension: int = 256):
        self.nvme = nvme_manager
        self.dimension = dimension

        # Memory-mapped arrays for persistent storage
        self.working_memory = self.nvme.create_memory_mapped_array(
            'working_memory', (100, dimension), np.float16
        )
        self.long_term_memory = self.nvme.create_memory_mapped_array(
            'long_term_memory', (10000, dimension), np.float16
        )
        self.emotional_memory = self.nvme.create_memory_mapped_array(
            'emotional_memory', (5000, dimension), np.float16
        )

        # Optimized data structures
        self.memory_indices = np.zeros(10000, dtype=np.int32)
        self.memory_timestamps = np.zeros(10000, dtype=np.float16)
        self.memory_access_counts = np.zeros(10000, dtype=np.int32)

        # Vectorized computation buffers
        self.similarity_buffer = np.zeros(1000, dtype=np.float16)
        self.relevance_buffer = np.zeros(1000, dtype=np.float16)

        logger.info("Optimized Memory System initialized with NVME storage")

    def vectorized_similarity_search(self, query: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized similarity search using optimized NumPy operations.
        Returns indices and similarities for top-k matches.
        """
        # Vectorized dot product computation
        similarities = np.dot(self.long_term_memory[:len(self.memory_indices)], query)

        # Get top-k indices using argpartition (faster than argsort for top-k)
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_similarities = similarities[top_indices]

        # Sort the top-k results
        sorted_order = np.argsort(top_similarities)[::-1]
        return top_indices[sorted_order], top_similarities[sorted_order]

    def batch_memory_consolidation(self, new_memories: np.ndarray, emotional_context: np.ndarray):
        """
        Batch process memory consolidation with vectorized operations.
        """
        batch_size = len(new_memories)

        # Vectorized emotional weighting
        emotional_weights = np.mean(np.abs(emotional_context), axis=1, keepdims=True)
        weighted_memories = new_memories * emotional_weights

        # Vectorized consolidation (mean pooling with decay)
        decay_factors = np.exp(-np.arange(batch_size, dtype=np.float16) / batch_size)
        consolidated = np.sum(weighted_memories * decay_factors[:, np.newaxis], axis=0) / np.sum(decay_factors)

        return consolidated

    def optimize_memory_layout(self):
        """
        Optimize memory layout by defragmenting and reorganizing arrays.
        """
        # Sort by access frequency for better cache performance
        sorted_indices = np.argsort(self.memory_access_counts)[::-1]

        # Reorganize arrays
        self.long_term_memory[:] = self.long_term_memory[sorted_indices]
        self.memory_timestamps[:] = self.memory_timestamps[sorted_indices]
        self.memory_access_counts[:] = self.memory_access_counts[sorted_indices]

        logger.info("Memory layout optimized for better performance")

    def store_consolidated_memory(self, consolidated_memory: np.ndarray, emotional_context: np.ndarray):
        """
        Store consolidated memory with emotional context in NVME storage.
        """
        # Find next available slot in long-term memory
        available_indices = np.where(self.memory_indices == 0)[0]
        if len(available_indices) > 0:
            slot = available_indices[0]
            self.long_term_memory[slot] = consolidated_memory
            self.memory_indices[slot] = 1
            self.memory_timestamps[slot] = time.time()
            self.memory_access_counts[slot] = 1

            # Store emotional context (handle dimension mismatch)
            emotional_slot = slot % self.emotional_memory.shape[0]
            if len(emotional_context) != self.emotional_memory.shape[1]:
                # Pad or truncate to match dimension
                if len(emotional_context) < self.emotional_memory.shape[1]:
                    padded_context = np.zeros(self.emotional_memory.shape[1])
                    padded_context[:len(emotional_context)] = emotional_context
                    self.emotional_memory[emotional_slot] = padded_context
                else:
                    self.emotional_memory[emotional_slot] = emotional_context[:self.emotional_memory.shape[1]]
            else:
                self.emotional_memory[emotional_slot] = emotional_context

            logger.debug(f"Stored consolidated memory at slot {slot}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'nvme_stats': self.nvme.get_memory_stats(),
            'memory_efficiency': len(self.memory_indices) / self.long_term_memory.shape[0],
            'average_access_count': np.mean(self.memory_access_counts),
            'memory_fragmentation': 1.0 - (np.count_nonzero(self.memory_indices) / len(self.memory_indices))
        }
