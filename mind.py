# mind.py

import pickle  # COMMENTED OUT: persistence logic
import numpy as np
import logging
logger = logging.getLogger(__name__)
from typing import Dict, List, Optional, Tuple, Any

from nn import Sequential, Linear, Sigmoid, ReLU, Tanh
from som import SelfOrganizingMap
from emotion import EmotionCore
from memory import MemorySystem
# from cafve import ACEConsciousnessTokenizer

# Dimension constants for import compatibility
SOM_ACTIVATION_DIM = 289   # 17x17 SOM activation dimension (prime number)
SOM_BMU_COORD_DIM = 2     # SOM BMU coordinate dimension (x,y)

class CognitiveScaling:
    """Manages dynamic dimensionality scaling across cognitive modules"""

    DIMENSION_STAGES = [
        (512, 1024, 2048),    # Initial stage (base, mid, high)
        (1024, 2048, 4096),   # First expansion
        (2048, 4096, 8192),   # Second expansion
        (4096, 8192, 16384)   # Final expansion tier
    ]

    def __init__(self, initial_stage: int = 0):
        self.current_stage = initial_stage
        self.base_dim, self.mid_dim, self.high_dim = self.DIMENSION_STAGES[initial_stage]
        self.growth_metrics = {
            'complexity_score': 0.0,
            'integration_score': 0.0,
            'stability_score': 0.0,
            'utilization_score': 0.0
        }

    def should_grow(self) -> bool:
        """Determine if cognitive dimensions should expand"""
        if self.current_stage >= len(self.DIMENSION_STAGES) - 1:
            return False

        # Weighted scoring for growth decision
        growth_score = (
            0.3 * self.growth_metrics['complexity_score'] +
            0.3 * self.growth_metrics['integration_score'] +
            0.2 * self.growth_metrics['stability_score'] +
            0.2 * self.growth_metrics['utilization_score']
        )

        return growth_score > 0.85

    def grow(self) -> bool:
        """Attempt to grow to next dimension stage"""
        if not self.should_grow():
            return False

        if self.current_stage < len(self.DIMENSION_STAGES) - 1:
            self.current_stage += 1
            self.base_dim, self.mid_dim, self.high_dim = self.DIMENSION_STAGES[self.current_stage]
            return True
        return False

    def update_metrics(self, complexity: float, integration: float,
                      stability: float, utilization: float):
        """Update growth metrics based on system performance"""
        self.growth_metrics['complexity_score'] = complexity
        self.growth_metrics['integration_score'] = integration
        self.growth_metrics['stability_score'] = stability
        self.growth_metrics['utilization_score'] = utilization

class CognitiveState:
    def __init__(self, dim: int):
        self.dimension = dim
        self.state_vector = np.zeros(dim, dtype=np.float32)
        self.emotional_influence = np.zeros(dim, dtype=np.float32)
        self.attention_mask = np.ones(dim, dtype=np.float32)
        self.uncertainty = np.zeros(dim, dtype=np.float32)

    def update(self, new_state: np.ndarray, emotional_context: np.ndarray,
              attention: np.ndarray, certainty: np.ndarray):
        self.state_vector = new_state
        self.emotional_influence = emotional_context
        self.attention_mask = attention
        self.uncertainty = 1 - certainty

    def get_weighted_state(self) -> np.ndarray:
        return self.state_vector * self.attention_mask * (1 - self.uncertainty)

    def get_confidence(self) -> float:
        """Calculate overall confidence in current cognitive state"""
        return 1.0 - np.mean(self.uncertainty)

    def get_attention_focus(self) -> np.ndarray:
        """Get the current attention distribution"""
        return self.attention_mask / (np.sum(self.attention_mask) + 1e-6)

class MetaCognition:
    """Handles self-reflection and cognitive monitoring"""
    def __init__(self, state_dim: int):
        self.state_dimension = state_dim
        self.reflection_network = Sequential(
            Linear(state_dim, state_dim * 2),
            Tanh(),
            Linear(state_dim * 2, state_dim),
            Sigmoid()
        )
        self.metacognitive_history = []

    def reflect(self, cognitive_state: CognitiveState) -> Tuple[np.ndarray, float]:
        """Analyze current cognitive state and generate insights"""
        state_vector = cognitive_state.get_weighted_state()
        reflection = self.reflection_network.forward(state_vector)
        coherence = cognitive_state.get_confidence() * np.mean(reflection)
        self.metacognitive_history.append((reflection, coherence))
        return reflection, coherence

# Module logger only: do not configure root logging handlers here.
# Logging configuration (handlers/format) should be centralized in `run.py` or the application entrypoint.
logger = logging.getLogger(__name__)

class Mind:
    """
    Core cognitive architecture coordinating SOM, Memory, Emotion, and more.
    Supports dynamic dimensional scaling for enhanced cognitive capacity.
    """
    def __init__(self, initial_dim_stage: int, som_activation_dim: int, som_bmu_coords_dim: int, 
                 emotional_state_dim: int, memory_recall_dim: int, predictive_error_dim: int, 
                 unified_cognitive_state_dim: int,
                 som_instance: Optional[SelfOrganizingMap] = None,
                 emotion_instance: Optional[EmotionCore] = None,
                 memory_instance: Optional[MemorySystem] = None):
        """
        Initializes the Mind module, which is the central cognitive processor.
        """
        logger.info(f"Initializing Mind module at stage {initial_dim_stage}...")
        
        # ADD THIS LINE TO CREATE THE SCALING OBJECT
        self.scaling = CognitiveScaling(initial_stage=initial_dim_stage)

        self.is_dreaming = False
        self.sensory_processing_enabled = True
        self.memory_replay_active = False

        # Store dimensions
        self.som_activation_dim = som_activation_dim
        self.som_bmu_coords_dim = som_bmu_coords_dim
        self.emotional_state_dim = emotional_state_dim
        self.memory_recall_dim = memory_recall_dim
        self.predictive_error_dim = predictive_error_dim
        self.unified_cognitive_state_dim = unified_cognitive_state_dim # ADDED THIS LINE

        # The total dimension of the concatenated input vector for the main network
        self.total_input_dim = (som_activation_dim + som_bmu_coords_dim + emotional_state_dim + 
                                memory_recall_dim + predictive_error_dim)

        self.cognitive_state = CognitiveState(self.scaling.base_dim)
        self.metacognition = MetaCognition(self.scaling.base_dim)

        # Initialize core modules with scalable dimensions
        # Dependency injection to avoid duplicate module instantiation during stabilization.
        # If external instances are provided we reuse them; otherwise we create new ones (legacy behavior).
        if emotion_instance is not None:
            self.emotion = emotion_instance
        else:
            self.emotion = EmotionCore(input_dim=self.scaling.base_dim)

        if memory_instance is not None:
            self.memory = memory_instance
        else:
            self.memory = MemorySystem(
                cognitive_state_dim=self.scaling.base_dim,
                emotional_state_dim=108  # Keep emotion dim fixed at 108
            )

        if som_instance is not None:
            self.som = som_instance
        else:
            self.som = SelfOrganizingMap(
                input_dim=self.scaling.base_dim,
                map_size=self._calculate_som_size()
            )

        # Integration networks
        internal_hidden_dim = 128
        self.integration_network = Sequential(
            Linear(self.total_input_dim, internal_hidden_dim),
            Tanh(),
            Linear(internal_hidden_dim, self.scaling.base_dim),
            Sigmoid()
        )

        # Internal state
        self._unified_cognitive_state = np.zeros(self.scaling.base_dim, dtype=np.float32)

        logger.info(f"Mind initialized with dynamic scaling. Current stage: {initial_dim_stage}")
        logger.info(f"Unified Cognitive State Dimension: {self.scaling.base_dim}")
        logger.info(f"Total Integrator Network Input Dimension: {self.total_input_dim}")

    def integrate_cognition(self,
                            som_activation_map: np.ndarray, # 17x17 map
                            som_bmu_coords: Tuple[int, int], # (row, col) of BMU
                            emotional_state: np.ndarray,    # 108D vector from EmotionCore
                            memory_recall_vector: np.ndarray, # 512D vector from AssociativeMemory
                            predictive_error_vector: np.ndarray, # 80D vector from PredictiveEngine
                            learning_bias_from_desire: float = 1.0, # From Desire module
                            attention_weights_from_attention: Optional[np.ndarray] = None # From Attention module
                            ) -> np.ndarray:
        """
        Processes and integrates various cognitive, emotional, and memory inputs
        into a unified cognitive state.
        """
        try:
            # 1. Flatten SOM activation map
            flat_som_activation = som_activation_map.flatten().astype(np.float32)

            # 2. Convert BMU coords to normalized vector (e.g., 0-1 range)
            # Assuming SOM map size is 17x17 for normalization
            normalized_bmu_coords = np.array([som_bmu_coords[0] / 16.0, som_bmu_coords[1] / 16.0], dtype=np.float32) # (17-1) for 0-indexed max

            # 3. Prepare Emotional State (already 108D)

            # 4. Prepare Memory Recall Vector (already 512D)

            # 5. Prepare Predictive Error Vector (already 80D)

            # --- Combine all inputs into a single vector ---
            parts = [
                flat_som_activation,
                normalized_bmu_coords,
                emotional_state,
                memory_recall_vector,
                predictive_error_vector
            ]
            combined_input = np.concatenate(parts).astype(np.float32)

            # Handle dimensional mismatches dynamically
            incoming_len = combined_input.shape[0]
            if incoming_len > self.total_input_dim:
                self._expand_integration_input(incoming_len)
            elif incoming_len < self.total_input_dim:
                pad_len = self.total_input_dim - incoming_len
                combined_input = np.concatenate([combined_input, np.zeros(pad_len, dtype=np.float32)])

            # Ensure no errors are thrown for mismatches
            if combined_input.shape[0] != self.total_input_dim:
                logger.warning(f"Post-adjustment length mismatch: {combined_input.shape[0]} vs {self.total_input_dim}. Using fallback state.")
                return self._unified_cognitive_state

            # Reshape for NN (batch_size=1)
            combined_input = combined_input.reshape(1, -1)

            # --- Apply attentional modulation (if Attention module provides weights) ---
            # This would apply attention_weights_from_attention as a gate or scalar multiplier
            # to parts of the combined_input or specific layers within the integrator_network.
            # For lean implementation, we'll assume it's an external influence on learning or processing.

            # --- Process through the Cognitive Integration Network ---
            new_unified_state = self.integration_network.forward(combined_input)

            # Update internal unified cognitive state
            self._unified_cognitive_state = new_unified_state[0, :] # Take first item from batch

            # Update cognitive state with current information
            self.cognitive_state.update(
                self._unified_cognitive_state,
                emotional_state[:self.scaling.base_dim] if len(emotional_state) > self.scaling.base_dim else emotional_state,
                np.ones(self.scaling.base_dim),
                np.ones(self.scaling.base_dim)
            )

            # --- Influence learning bias (e.g., for SOM, Memory) ---
            # The 'learning_bias_from_desire' directly scales the learning rate
            # for modules like SOM and AssociativeMemory during their learn step.
            # This module doesn't apply it, but produces the current state influenced by it.

            logger.info("Unified cognitive state updated successfully.")
            return self._unified_cognitive_state.copy()
        except Exception as e:
            logger.error(f"Error in integrate_cognition: {e}")
            return self._unified_cognitive_state.copy()

    def get_current_dimensions(self) -> Dict[str, int]:
        return {
            'base_dim': self.scaling.base_dim,
            'mid_dim': self.scaling.mid_dim,
            'high_dim': self.scaling.high_dim,
            'integrator_input_dim': self.total_input_dim
        }

    def _expand_integration_input(self, new_total_input_dim: int):
        """Expand the first layer of the integration network to accept larger input.
        Copies existing weights; new columns are Xavier-initialized. Adjusts total_input_dim.
        """
        if new_total_input_dim <= self.total_input_dim:
            return
        try:
            first_layer = self.integration_network.layers[0]
            if not isinstance(first_layer, Linear):
                logger.warning("First layer of integration_network not Linear; skipping expansion.")
                return
            old_W = first_layer.weights  # shape (in_dim, out_dim)
            old_in = old_W.shape[0]
            out_dim = old_W.shape[1]
            # Xavier init for new rows (additional input features)
            import math
            new_rows = new_total_input_dim - old_in
            limit = math.sqrt(6.0 / (new_total_input_dim + out_dim))
            extra_W = np.random.uniform(-limit, limit, (new_rows, out_dim)).astype(old_W.dtype)
            new_W = np.concatenate([old_W, extra_W], axis=0)
            first_layer.weights = new_W
            # Biases unchanged (shape (1, out_dim))
            self.total_input_dim = new_total_input_dim
            logger.info(f"Integration input dimension expanded to {new_total_input_dim}.")
        except Exception as e:
            logger.warning(f"Failed to expand integration input dimension: {e}")

    def get_unified_cognitive_state(self) -> np.ndarray:
        """Returns the current unified cognitive state vector."""
        return self._unified_cognitive_state.copy()

    # --- Backward Compatibility Adapter ---
    def process_cognitive_state(self, som_activation_flat: np.ndarray, sensory_data: dict) -> np.ndarray:
        """Legacy method expected by main.py

        The newer architecture uses integrate_cognition with richer inputs. This
        adapter reconstructs minimal placeholders so existing call sites work.
        som_activation_flat: Flattened SOM activation (may already be 1D)
        sensory_data: dict with any available context (ignored for now)
        Returns current unified cognitive state (updated).
        """
        try:
            # Validate input
            logger.debug(f"Received som_activation_flat: {som_activation_flat}")
            logger.debug(f"Received sensory_data: {sensory_data}")
            if som_activation_flat is None:
                logger.error("Mind.process_cognitive_state received None for som_activation_flat. Investigating upstream data flow.")
                return self._unified_cognitive_state.copy()

            # Rebuild square map if possible
            side = int(np.sqrt(som_activation_flat.shape[0]))
            if side * side == som_activation_flat.shape[0] and side > 0:
                som_map = som_activation_flat.reshape(side, side)
                logger.debug(f"Rebuilt som_map: {som_map}")
            else:
                # Fallback: log and skip processing
                logger.error("Mind.process_cognitive_state received invalid som_activation_flat dimensions. Expected square dimensions.")
                return self._unified_cognitive_state.copy()

            # Placeholder BMU coords: pick max activation
            max_idx = np.argmax(som_map)
            bmu_coords = (max_idx // som_map.shape[1], max_idx % som_map.shape[1])
            logger.debug(f"Calculated BMU coordinates: {bmu_coords}")

            # Placeholder emotional state, memory recall, predictive error (standardized large dims)
            emotional_state = np.zeros(self.emotional_state_dim, dtype=np.float32)
            memory_recall = np.zeros(self.memory_recall_dim, dtype=np.float32)
            predictive_error = np.zeros(self.predictive_error_dim, dtype=np.float32)

            unified = self.integrate_cognition(
                som_map,
                bmu_coords,
                emotional_state,
                memory_recall,
                predictive_error
            )
            logger.debug(f"Unified cognitive state: {unified}")
            return unified
        except Exception as e:
            logger.exception(f"Exception in Mind.process_cognitive_state: {e}")
            return self._unified_cognitive_state.copy()

    def get_current_dimensions(self) -> Dict[str, int]:
        """Get current dimension configuration"""
        return {
            'base_dim': self.scaling.base_dim,
            'mid_dim': self.scaling.mid_dim,
            'high_dim': self.scaling.high_dim,
            'stage': self.scaling.current_stage
        }

    def attempt_growth(self) -> bool:
        """Attempt to grow cognitive dimensions based on current metrics"""
        # Calculate current metrics
        complexity = self._calculate_complexity_score()
        integration = self._calculate_integration_score()
        stability = self._calculate_stability_score()
        utilization = self._calculate_utilization_score()

        # Update scaling metrics
        self.scaling.update_metrics(complexity, integration, stability, utilization)

        # Attempt growth
        if self.scaling.grow():
            self._grow_cognitive_dimensions()
            return True
        return False

    def get_networks(self) -> List[None]:
        """Returns a list of all internal neural networks for optimization."""
        return [self.integration_network, self.metacognition.reflection_network]

    def _calculate_som_size(self) -> Tuple[int, int]:
        """Calculate SOM map size using prime numbers based on current base dimension"""
        # Use prime numbers for SOM map size (user requirement)
        primes = [17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

        # Find appropriate prime based on current dimension stage
        stage = self.scaling.current_stage
        if stage < len(primes):
            size = primes[stage]
        else:
            # For stages beyond our prime list, use the largest prime
            size = primes[-1]

        return (size, size)

    def _calculate_complexity_score(self) -> float:
        """Calculate system complexity based on state variance and network activity"""
        if not hasattr(self, '_complexity_history'):
            self._complexity_history = []

        # Measure state complexity using variance and entropy
        state_variance = np.var(self._unified_cognitive_state)
        state_entropy = -np.sum(self._unified_cognitive_state * np.log(self._unified_cognitive_state + 1e-10))
        state_entropy = np.clip(state_entropy, 0, 100)  # Prevent extreme values

        # Network activity based on parameter magnitudes
        total_params = 0
        param_magnitude = 0
        for param, _, _ in self.integration_network.get_trainable_params():
            total_params += param.size
            param_magnitude += np.sum(np.abs(param))

        network_activity = param_magnitude / (total_params + 1e-6)

        complexity = (0.4 * state_variance + 0.3 * state_entropy + 0.3 * network_activity)
        self._complexity_history.append(complexity)

        if len(self._complexity_history) > 50:
            self._complexity_history.pop(0)

        return np.mean(self._complexity_history)

    def _calculate_stability_score(self) -> float:
        """Calculate system stability based on recent state changes"""
        if not hasattr(self, '_state_history'):
            self._state_history = []
            return 1.0

        self._state_history.append(self.cognitive_state.state_vector.copy())
        if len(self._state_history) > 100:
            self._state_history.pop(0)

        if len(self._state_history) < 2:
            return 1.0

        # Calculate average state change
        changes = [np.mean(np.abs(self._state_history[i] - self._state_history[i-1]))
                  for i in range(1, len(self._state_history))]

        # Convert to stability score (1 = very stable, 0 = very unstable)
        avg_change = np.mean(changes)
        stability = 1.0 / (1.0 + avg_change)
        return stability

    def _calculate_integration_score(self) -> float:
        """Calculate how well different cognitive components are integrated"""
        # Get current states
        cognitive = self.cognitive_state.state_vector
        emotional = self.cognitive_state.emotional_influence
        attention = self.cognitive_state.attention_mask

        # Calculate correlations between components
        corr_cog_emo = np.corrcoef(cognitive, emotional)[0,1]
        corr_cog_att = np.corrcoef(cognitive, attention)[0,1]
        corr_emo_att = np.corrcoef(emotional, attention)[0,1]

        # Average the absolute correlations
        integration = np.mean([np.abs(corr_cog_emo),
                             np.abs(corr_cog_att),
                             np.abs(corr_emo_att)])

        return float(integration)

    def _calculate_utilization_score(self) -> float:
        """Calculate how effectively the cognitive capacity is being used"""
        # Measure non-zero activation across the state vector
        utilization = np.mean(np.abs(self._unified_cognitive_state) > 0.01)
        return float(utilization)

    def _grow_cognitive_dimensions(self) -> None:
        """Handle growth of all cognitive dimensions"""
        old_base_dim = self.scaling.base_dim
        old_mid_dim = self.scaling.mid_dim
        old_high_dim = self.scaling.high_dim

        # Get new dimensions (already updated in scaling.grow())
        new_base_dim = self.scaling.base_dim
        new_mid_dim = self.scaling.mid_dim
        new_high_dim = self.scaling.high_dim

        # Scale up cognitive state
        new_cognitive_state = CognitiveState(new_base_dim)
        new_cognitive_state.state_vector[:old_base_dim] = self.cognitive_state.state_vector
        new_cognitive_state.emotional_influence[:old_base_dim] = self.cognitive_state.emotional_influence
        new_cognitive_state.attention_mask[:old_base_dim] = self.cognitive_state.attention_mask
        new_cognitive_state.uncertainty[:old_base_dim] = self.cognitive_state.uncertainty
        self.cognitive_state = new_cognitive_state

        # Update unified cognitive state array
        new_unified_state = np.zeros(new_base_dim, dtype=np.float32)
        new_unified_state[:old_base_dim] = self._unified_cognitive_state
        self._unified_cognitive_state = new_unified_state

        # Scale up emotion system
        self.emotion = EmotionCore(
            input_dim=new_base_dim
        )

        # Scale up memory system
        new_memory = MemorySystem(
            cognitive_state_dim=new_base_dim,
            emotional_state_dim=108  # Keep emotion dim fixed at 108
        )
        new_memory.transfer_memories(self.memory)
        self.memory = new_memory

        # Update SOM
        new_som_size = self._calculate_som_size()
        self.som = SelfOrganizingMap(
            input_dim=new_base_dim,
            map_size=new_som_size
        )

        # Update integration network
        self.integration_network = Sequential(
            Linear(self.total_input_dim, new_mid_dim),
            Tanh(),
            Linear(new_mid_dim, new_base_dim),
            Sigmoid()
        )

        # Update metacognition
        self.metacognition = MetaCognition(new_base_dim)

        # Update dimension tracking
        self.unified_cognitive_state_dim = new_base_dim

        logger.info(f"✅ Cognitive dimensions grown: {old_base_dim}->{new_base_dim}, "
                   f"{old_mid_dim}->{new_mid_dim}, {old_high_dim}->{new_high_dim}")

    # Persistence methods (save/load state)
    # def save_state(self, save_path: str):
    #     """Saves the Mind's state (unified cognitive state and integrator network weights) to a file."""
    #     try:
    #         state = {
    #             'unified_cognitive_state': self._unified_cognitive_state.tolist(),
    #             'scaling_stage': self.scaling.current_stage,
    #             'cognitive_state': {
    #                 # 'state_vector': self.cognitive_state.state_vector.tolist(),
    #                 # 'emotional_influence': self.cognitive_state.emotional_influence.tolist(),
    #                 # 'attention_mask': self.cognitive_state.attention_mask.tolist(),
    #                 # 'uncertainty': self.cognitive_state.uncertainty.tolist()
    #             },
    #             # Save integrator network weights
    #             # 'integrator_network_weights': [(p[0].tolist(), p[1]) for p in self.integration_network.get_trainable_params()],
    #             # 'metacognition_weights': [(p[0].tolist(), p[1]) for p in self.metacognition.reflection_network.get_trainable_params()]
    #         }
    #         with open(save_path, 'wb') as f:
    #             pickle.dump(state, f)
    #         logger.info(f"Mind state saved to {save_path}")
    #     except Exception as e:
    #         logger.error(f"Error saving Mind state: {e}")

    # def load_state(self, load_path: str):
    # """Loads the Mind's state from a file."""
    # try:
    #     with open(load_path, 'rb') as f:
    #         state = pickle.load(f)
    #
    #     # Load scaling stage
    #     # if 'scaling_stage' in state:
    #         self.scaling.current_stage = state['scaling_stage']
    #         self.scaling.base_dim, self.scaling.mid_dim, self.scaling.high_dim = self.scaling.DIMENSION_STAGES[self.scaling.current_stage]
    #
    #     # Load unified cognitive state
    #     # if np.array(state['unified_cognitive_state']).shape == self._unified_cognitive_state.shape:
    #         self._unified_cognitive_state = np.array(state['unified_cognitive_state'], dtype=np.float32)
    #
    #         # Load cognitive state
    #         if 'cognitive_state' in state:
    #             cog_state = state['cognitive_state']
    #             self.cognitive_state.state_vector = np.array(cog_state['state_vector'], dtype=np.float32)
    #             self.cognitive_state.emotional_influence = np.array(cog_state['emotional_influence'], dtype=np.float32)
    #             self.cognitive_state.attention_mask = np.array(cog_state['attention_mask'], dtype=np.float32)
    #             self.cognitive_state.uncertainty = np.array(cog_state['uncertainty'], dtype=np.float32)
    #
    #         # Load neural network weights
    #         # loaded_params = state.get('integrator_network_weights', [])
    #         # current_params = self.integration_network.get_trainable_params()
    #
    #
    #         # Load metacognition weights
    #         # meta_params = state.get('metacognition_weights', [])
    #         # meta_current = self.metacognition.reflection_network.get_trainable_params()
    #
    #
    ##                logger.info(f"Mind state loaded from {load_path}")
    # #           else:
    #         logger.warning("Loaded Mind state dimensions mismatch. Initializing to default.")
    #         self._unified_cognitive_state = np.zeros(self.unified_cognitive_state_dim, dtype=np.float32)
    #
    # except FileNotFoundError:
    #     logger.warning(f"Mind state file not found at {load_path}. Initializing to default.")
    #     self._unified_cognitive_state = np.zeros(self.unified_cognitive_state_dim, dtype=np.float32)
    # except Exception as e:
    #     logger.error(f"Error loading Mind state: {e}. Initializing to default.")
    #     self._unified_cognitive_state = np.zeros(self.unified_cognitive_state_dim, dtype=np.float32)


# Test block (can be removed in final deployment)
if __name__ == "__main__":
    logger.info("Mind module loaded successfully.")
