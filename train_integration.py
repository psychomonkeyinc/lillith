# train_integration.py
# Integration module connecting neural fabric training with LILLITH's sensory pipeline

import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

from training import NeuralFabricTrainer, TrainingConfig
from cafve import ConsciousnessAwareFeatureVectorEncoder
from emotion import EmotionCore
from memory import MemorySystem
from goals import Goals

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LillithTrainingOrchestrator:
    """
    Orchestrates neural fabric training with LILLITH's sensory and cognitive systems.
    Enables continuous online learning from real sensory input.
    """
    def __init__(self, 
                 initial_cog_state_dim: int = 512,
                 emotion_dim: int = 512,
                 som_map_size: Tuple[int, int] = (17, 17),
                 enable_continuous_learning: bool = True):
        
        self.initial_cog_state_dim = initial_cog_state_dim
        self.emotion_dim = emotion_dim
        self.som_map_size = som_map_size
        self.enable_continuous_learning = enable_continuous_learning
        
        # Configure training system
        self.training_config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=4,
            time_steps=16,
            spatial_dims=(som_map_size[0], som_map_size[1], 64),
            use_random_prop=True,
            use_intentional_prop=True,
            bidirectional_weight=0.5,
            temporal_coherence_weight=0.3,
            dendritic_integration=True,
            max_epochs=None  # Continuous learning
        )
        
        # Initialize trainer
        self.trainer = NeuralFabricTrainer(self.training_config)
        
        # Sensory buffer for temporal batching
        self.sensory_buffer = deque(maxlen=self.training_config.batch_size * 2)
        
        # Training statistics
        self.total_updates = 0
        self.last_train_time = time.time()
        self.train_interval = 0.1  # Train every 100ms
        
        # Mode cycling
        self.mode_counter = 0
        self.modes = ['normal', 'random', 'intentional']
        
        logger.info("Lillith Training Orchestrator initialized")
        logger.info(f"  Continuous learning: {enable_continuous_learning}")
        logger.info(f"  Spatial dims: {self.training_config.spatial_dims}")
        logger.info(f"  Batch size: {self.training_config.batch_size}")
    
    def process_sensory_input(self, 
                             visual_features: np.ndarray,
                             audio_features: np.ndarray,
                             cognitive_state: np.ndarray,
                             emotion_state: np.ndarray) -> Dict[str, Any]:
        """
        Process multi-modal sensory input and perform online learning.
        
        Args:
            visual_features: Visual sensory features
            audio_features: Audio sensory features
            cognitive_state: Current cognitive state
            emotion_state: Current emotional state
            
        Returns:
            Dictionary with training metrics and learned representations
        """
        # Combine multi-modal features
        combined_features = self._combine_features(
            visual_features, audio_features, cognitive_state, emotion_state
        )
        
        # Add to buffer
        self.sensory_buffer.append(combined_features)
        
        # Check if we should train
        current_time = time.time()
        should_train = (
            self.enable_continuous_learning and
            len(self.sensory_buffer) >= self.training_config.batch_size and
            (current_time - self.last_train_time) >= self.train_interval
        )
        
        if should_train:
            metrics = self._train_on_buffer()
            self.last_train_time = current_time
            return metrics
        else:
            # Just do a forward pass for inference
            return self._forward_inference(combined_features)
    
    def _combine_features(self, visual: np.ndarray, audio: np.ndarray,
                         cognitive: np.ndarray, emotion: np.ndarray) -> np.ndarray:
        """Combine multi-modal features into unified representation"""
        # Ensure all features have compatible sizes
        target_size = np.prod(self.training_config.spatial_dims)
        
        def resize_feature(feat: np.ndarray, target: int) -> np.ndarray:
            if feat.size == 0:
                return np.zeros(target, dtype=np.float32)
            flat = feat.flatten()
            if flat.size > target:
                return flat[:target]
            elif flat.size < target:
                return np.pad(flat, (0, target - flat.size))
            return flat
        
        # Resize each modality
        visual_resized = resize_feature(visual, target_size // 4)
        audio_resized = resize_feature(audio, target_size // 4)
        cognitive_resized = resize_feature(cognitive, target_size // 4)
        emotion_resized = resize_feature(emotion, target_size // 4)
        
        # Concatenate
        combined = np.concatenate([
            visual_resized, audio_resized, cognitive_resized, emotion_resized
        ])
        
        # Reshape to 3D spatial structure
        combined = combined.reshape(self.training_config.spatial_dims)
        
        return combined
    
    def _train_on_buffer(self) -> Dict[str, Any]:
        """Train on accumulated sensory buffer"""
        # Extract batch from buffer
        batch_data = list(self.sensory_buffer)[-self.training_config.batch_size:]
        batch = np.stack(batch_data)
        
        # Add channel dimension if needed
        if len(batch.shape) == 4:  # (batch, x, y, z)
            batch = np.expand_dims(batch, axis=-1)  # (batch, x, y, z, 1)
        
        # Select training mode
        mode = self.modes[self.mode_counter % len(self.modes)]
        self.mode_counter += 1
        
        # Perform training step
        metrics = self.trainer.train_step(batch, mode=mode)
        
        self.total_updates += 1
        
        # Log progress periodically
        if self.total_updates % 100 == 0:
            logger.info(f"Training update {self.total_updates}: "
                       f"loss={metrics.get('loss', 0):.6f}, "
                       f"mode={mode}, energy={metrics.get('energy', 0):.3f}")
        
        metrics['mode'] = mode
        metrics['total_updates'] = self.total_updates
        metrics['buffer_size'] = len(self.sensory_buffer)
        
        return metrics
    
    def _forward_inference(self, features: np.ndarray) -> Dict[str, Any]:
        """Perform forward inference without training"""
        # Add batch dimension
        batch = np.expand_dims(features, axis=0)
        
        # Add channel dimension
        if len(batch.shape) == 4:
            batch = np.expand_dims(batch, axis=-1)
        
        # Forward pass
        outputs = self.trainer.forward_pass(batch, mode='normal')
        
        return {
            'som_activation': outputs['som_output'],
            'fabric_activation': outputs['fabric_output'],
            'inference_only': True
        }
    
    def set_learning_mode(self, enabled: bool):
        """Enable or disable continuous learning"""
        self.enable_continuous_learning = enabled
        logger.info(f"Continuous learning {'enabled' if enabled else 'disabled'}")
    
    def save_training_state(self, filepath: str):
        """Save current training state"""
        state = self.trainer.get_state_dict()
        state['orchestrator'] = {
            'total_updates': self.total_updates,
            'mode_counter': self.mode_counter,
            'enable_continuous_learning': self.enable_continuous_learning
        }
        
        np.save(filepath, state)
        logger.info(f"Training state saved to {filepath}")
    
    def load_training_state(self, filepath: str):
        """Load training state"""
        state = np.load(filepath, allow_pickle=True).item()
        
        self.trainer.load_state_dict(state)
        
        if 'orchestrator' in state:
            orch_state = state['orchestrator']
            self.total_updates = orch_state.get('total_updates', 0)
            self.mode_counter = orch_state.get('mode_counter', 0)
            self.enable_continuous_learning = orch_state.get('enable_continuous_learning', True)
        
        logger.info(f"Training state loaded from {filepath}")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get current training statistics"""
        return {
            'total_updates': self.total_updates,
            'buffer_size': len(self.sensory_buffer),
            'continuous_learning': self.enable_continuous_learning,
            'current_mode': self.modes[self.mode_counter % len(self.modes)],
            'trainer_epoch': self.trainer.epoch,
            'trainer_steps': self.trainer.total_steps,
            'recent_losses': list(self.trainer.loss_history)[-10:] if self.trainer.loss_history else []
        }


def integrate_with_main_loop(orchestrator: LillithTrainingOrchestrator,
                            cafve_encoder: Optional[Any] = None,
                            emotion_core: Optional[Any] = None,
                            goals_system: Optional[Any] = None) -> callable:
    """
    Create an integration function for LILLITH's main processing loop.
    
    Returns:
        A function that can be called in the main loop to process sensory data
        and perform online learning.
    """
    def process_cycle(visual_input: np.ndarray,
                     audio_input: np.ndarray,
                     cognitive_state: np.ndarray) -> Dict[str, Any]:
        """
        Process one cycle of sensory input with training.
        """
        # Extract emotion state if available
        if emotion_core is not None:
            emotion_state = emotion_core.get_emotion_vector()
        else:
            emotion_state = np.zeros(512, dtype=np.float32)
        
        # Check if learning should be modulated by goals
        if goals_system is not None:
            learning_drive = goals_system.calculate_learning_bias()
            # Adjust training based on learning drive
            orchestrator.training_config.learning_rate = 1e-4 * learning_drive
        
        # Process through training system
        metrics = orchestrator.process_sensory_input(
            visual_features=visual_input,
            audio_features=audio_input,
            cognitive_state=cognitive_state,
            emotion_state=emotion_state
        )
        
        return metrics
    
    return process_cycle


# Example usage and testing
if __name__ == '__main__':
    logger.info("Testing Lillith Training Integration")
    
    # Create orchestrator
    orchestrator = LillithTrainingOrchestrator(
        initial_cog_state_dim=512,
        emotion_dim=512,
        som_map_size=(13, 13),
        enable_continuous_learning=True
    )
    
    # Simulate sensory inputs
    for i in range(10):
        visual = np.random.randn(128).astype(np.float32)
        audio = np.random.randn(128).astype(np.float32)
        cognitive = np.random.randn(512).astype(np.float32)
        emotion = np.random.randn(512).astype(np.float32)
        
        metrics = orchestrator.process_sensory_input(
            visual, audio, cognitive, emotion
        )
        
        if metrics:
            logger.info(f"Step {i}: {metrics.get('mode', 'inference')} - "
                       f"loss={metrics.get('loss', 0):.6f}")
    
    # Get statistics
    stats = orchestrator.get_training_statistics()
    logger.info(f"Training statistics: {stats}")
    
    logger.info("Integration test complete")
