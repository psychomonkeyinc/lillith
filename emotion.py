# emotion.py

import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from nn import Sequential, Linear, Sigmoid, ReLU, Tanh

# Single consolidated state file for all modules

class EmotionalState:
    """Represents a complex emotional state with multiple dimensions"""
    def __init__(self, base_dimension: int):
        self.dimension = base_dimension
        self.valence = np.zeros(base_dimension)  # Positive/negative
        self.arousal = np.zeros(base_dimension)  # Energy level
        self.dominance = np.zeros(base_dimension)  # Control/influence
        self.intensity = np.zeros(base_dimension)  # Strength of emotion
        self.temporal_context = []  # Historical emotional context
        
    def update(self, new_valence: np.ndarray, new_arousal: np.ndarray, 
              new_dominance: np.ndarray, new_intensity: np.ndarray):
        self.valence = new_valence
        self.arousal = new_arousal
        self.dominance = new_dominance
        self.intensity = new_intensity
        self.temporal_context.append((time.time(), self.get_unified_state()))
        if len(self.temporal_context) > 100:  # Keep last 100 states
            self.temporal_context.pop(0)
            
    def get_unified_state(self) -> np.ndarray:
        """Combine all emotional dimensions into a unified representation"""
        return np.stack([
            self.valence * self.intensity,
            self.arousal * self.intensity,
            self.dominance * self.intensity
        ]).mean(axis=0)

class EmotionalModulation:
    """Handles how emotions modulate cognitive processes"""
    def __init__(self, emotion_dim: int, cognitive_dim: int):
        self.emotion_dimension = emotion_dim
        self.cognitive_dimension = cognitive_dim
        self.modulation_network = Sequential(
            Linear(emotion_dim, (emotion_dim + cognitive_dim) // 2),
            Tanh(),
            Linear((emotion_dim + cognitive_dim) // 2, cognitive_dim),
            Sigmoid()
        )
        
    def compute_modulation(self, emotional_state: np.ndarray) -> np.ndarray:
        """Compute cognitive modulation factors based on emotional state"""
        return self.modulation_network.forward(emotional_state)
        
    def get_emotional_influence(self, emotional_state: np.ndarray) -> float:
        """Calculate overall emotional influence strength"""
        return np.mean(np.abs(emotional_state))


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmotionCore:
    """
    The Heart: Manages Lillith's 108-dimensional emotional state.
    Allows for blending, temporal decay, and influences persona based on intensity.
    Input: CAFVE tokens (~80D). Output: 108D emotional state vector.
    """
    
    # Define the 108 emotional dimensions
    # Ordered for consistency in indexing
    EMOTION_DIMENSIONS = [
        # Core Primary (7)
        #'Happy', 'Sad', 'Disgusted', 'Angry', 'Fearful', 'Surprised', 'Good',
        # Pleasure-Based / Good Spinoffs (13)'Euphoria', 'Eureka', 'Sexual_Climax', 'Ecstasy', 'Bliss', 'Arousal_Positive', 
        'Warmth', 'Satiation', 'Contentment_Deep', 'Thrill', 'Admiration', 'Harmony', 'Prideful',
        # Other Wheel Spin-offs (88 unique from the exhaustive list provided previously)
        # These represent nuances, intensities, or blends, allowing for 34k nuances in this 108D space.
        'Bad', 'Startled', 'Confused', 'Amazed', 'Excited', 'Playful', 'Content', 
        'Interested', 'Proud', 'Accepted', 'Powerful', 'Peaceful', 'Trusting', 'Optimistic', 
        'Loving', 'Thankful', 'Sensitive', 'Intimate', 'Hopeful', 'Inspired', 'Joyful', 
        'Curious', 'Inquisitive', 'Successful', 'Confident', 'Respected', 'Valued', 
        'Courageous', 'Creative', 'Energetic', 'Aroused', 'Cheeky', 'Free', 'Eager', 'Awe', 
        'Astonished', 'Perplexed', 'Overwhelmed', 'Out_of_control', 'Unfocused', 'Sleepy', 
        'Rushed', 'Pressured', 'Apathetic', 'Indifferent', 'Helpless', 'Frightened', 'Worried', 
        'Inadequate', 'Inferior', 'Worthless', 'Insignificant', 'Excluded', 'Persecuted', 
        'Nervous', 'Exposed', 'Betrayed', 'Resentful', 'Disrespected', 'Ridiculed', 
        'Indignant', 'Violated', 'Furious', 'Jealous', 'Provoked', 'Hostile', 'Infuriated', 
        'Annoyed', 'Withdrawn', 'Numb', 'Sceptical', 'Dismissive', 'Judgmental', 'Embarrassed', 
        'Appalled', 'Revolted', 'Nauseated', 'Detestable', 'Horrified', 'Hesitant', 
        'Lonely', 'Vulnerable', 'Despair', 'Guilty', 'Depressed', 'Hurt', 'Repelled', 'Awful'
    ]
    
    # Global mapping of emotion name to its index in the vector
    EMOTION_NAME_TO_INDEX = {name: i for i, name in enumerate(EMOTION_DIMENSIONS)}

    # Categorize emotions for differential decay rates
    # These sets are based on the full 108-dimension list provided
    POSITIVE_SET = {
        'Happy', 'Good', 'Euphoria', 'Eureka', 'Sexual_Climax', 'Ecstasy', 'Bliss', 
        'Arousal_Positive', 'Warmth', 'Satiation', 'Contentment_Deep', 'Thrill', 
        'Admiration', 'Harmony', 'Prideful', 'Playful', 'Content', 'Interested', 
        'Proud', 'Accepted', 'Powerful', 'Peaceful', 'Trusting', 'Optimistic', 
        'Loving', 'Thankful', 'Sensitive', 'Intimate', 'Hopeful', 'Inspired', 
        'Joyful', 'Curious', 'Inquisitive', 'Successful', 'Confident', 'Respected', 
        'Valued', 'Courageous', 'Creative', 'Energetic', 'Awe', 'Eager' # 42 emotions
    } 
    NEGATIVE_SET = {
        'Sad', 'Disgusted', 'Angry', 'Fearful', 'Lonely', 'Vulnerable', 'Despair', 
        'Guilty', 'Depressed', 'Hurt', 'Repelled', 'Awful', 'Disappointed', 'Disapproving', 
        'Critical', 'Distant', 'Frustrated', 'Aggressive', 'Mad', 'Bitter', 'Humiliated', 
        'Let down', 'Threatened', 'Rejected', 'Insecure', 'Weak', 'Anxious', 'Scared', 
        'Bored', 'Busy', 'Stressed', 'Tired', 'Overwhelmed', 'Out_of_control', 'Unfocused', 
        'Sleepy', 'Rushed', 'Pressured', 'Apathetic', 'Indifferent', 'Helpless', 
        'Frightened', 'Worried', 'Inadequate', 'Inferior', 'Worthless', 'Insignificant', 
        'Excluded', 'Persecuted', 'Nervous', 'Exposed', 'Betrayed', 'Resentful', 
        'Disrespected', 'Ridiculed', 'Indignant', 'Violated', 'Furious', 'Jealous', 
        'Provoked', 'Hostile', 'Infuriated', 'Annoyed', 'Withdrawn', 'Numb', 'Sceptical', 
        'Dismissive', 'Judgmental', 'Embarrassed', 'Appalled', 'Revolted', 'Nauseated', 
        'Detestable', 'Horrified', 'Hesitant' # 60 emotions
    } 
    # The remaining emotions are Neutral/Complex:
    # 'Bad', busy"Good",'Startled', 'Confused', 'Amazed', 'Excited', 'Perplexed'

    def __init__(self, input_dim: int = 80,
                 output_dim: int = 512,
                 internal_hidden_dim: int = 128,
                 decay_rate_good: float = 0.01, # Slower decay for positive
                 decay_rate_bad: float = 0.03,  # Faster decay for negative
                 decay_rate_neutral: float = 0.02, # Moderate decay for neutral/complex
                 persona_outlier_threshold: float = 0.2, # Below this intensity, emotions don't impact persona
                 blending_factor: float = 0.3): # How much new input influences current state
        
        self.output_dim = output_dim  # Use dynamic dimension instead of hardcoded 108
        self.input_dim = input_dim # CAFVE token input
        
        # Current emotional state (intensities for each dimension, values 0-1)
        self.emotional_state = np.zeros(self.output_dim, dtype=np.float32)
        # Timestamp of last update for each emotion, for individual decay calculation
        self.last_update_times = np.full(self.output_dim, time.perf_counter(), dtype=np.float32)

        # Map each emotion index to its decay type ('POSITIVE', 'NEGATIVE', 'NEUTRAL_COMPLEX')
        self._emotion_types = {} 
        for i, emotion_name in enumerate(self.EMOTION_DIMENSIONS):
            if emotion_name in self.POSITIVE_SET:
                self._emotion_types[i] = 'POSITIVE'
            elif emotion_name in self.NEGATIVE_SET:
                self._emotion_types[i] = 'NEGATIVE'
            else: # Anything not explicitly positive or negative from the sets
                self._emotion_types[i] = 'NEUTRAL_COMPLEX'
        # For any additional emergent dimensions beyond the named list, default to NEUTRAL_COMPLEX
        if self.output_dim > len(self.EMOTION_DIMENSIONS):
            for i in range(len(self.EMOTION_DIMENSIONS), self.output_dim):
                self._emotion_types[i] = 'NEUTRAL_COMPLEX'
        
        self.decay_rates = {
            'POSITIVE': np.float32(decay_rate_good),
            'NEGATIVE': np.float32(decay_rate_bad),
            'NEUTRAL_COMPLEX': np.float32(decay_rate_neutral)
        }
        self.persona_outlier_threshold = np.float32(persona_outlier_threshold)
        self.blending_factor = np.float32(blending_factor)

        # Neural network to map input (CAFVE tokens) to 108D emotional state
        # Uses Sequential, Linear, ReLU, Sigmoid from nn.py
        self.emotional_mapper = Sequential(
            Linear(self.input_dim, internal_hidden_dim),
            ReLU(),
            Linear(internal_hidden_dim, internal_hidden_dim),
            ReLU(),
            Linear(internal_hidden_dim, self.output_dim),
            Sigmoid()
        )
        
        logger.info(f"EmotionCore initialized with {self.output_dim} dimensions. Input_dim: {self.input_dim}")

        # Track a combined dimension attribute for downstream dimension audits
        self.combined_dim = self.output_dim

    def _apply_temporal_decay(self):
        """Applies temporal decay to each emotional dimension based on its category."""
        current_time = time.perf_counter()
        
        for i in range(self.output_dim):
            # Calculate time elapsed since last update for this specific emotion
            time_elapsed_for_this_emotion = current_time - self.last_update_times[i]
            
            decay_rate = self.decay_rates[self._emotion_types[i]]
            
            # Apply decay only if intensity is positive
            if self.emotional_state[i] > 0.0:
                # decay_factor = exp(-rate * time_elapsed)
                decay_factor = np.exp(-decay_rate * time_elapsed_for_this_emotion)
                self.emotional_state[i] = self.emotional_state[i] * decay_factor
                # Ensure intensity doesn't go below zero due to float precision
                self.emotional_state[i] = np.clip(self.emotional_state[i], 0.0, 1.0)
            
            # Update last_update_time for this specific emotion
            self.last_update_times[i] = current_time

    def process_input(self, cafve_token_batch: List[np.ndarray]) -> np.ndarray:
        """
        Processes a batch of CAFVE tokens, updates emotional state, and applies decay.
        Returns the current 108-dimensional emotional state vector.
        """
        # Apply decay based on time elapsed since the last full processing cycle
        self._apply_temporal_decay() 

        if not cafve_token_batch:
            # If no new input, just return the decayed state
            return self.emotional_state.copy()

        # Concatenate tokens for batch processing by the emotional_mapper NN
        # Ensure batch_size x input_dim
        batch_input = np.array(cafve_token_batch, dtype=np.float32)
        
        # Predict new emotional intensities based on input
        predicted_intensities_batch = self.emotional_mapper.forward(batch_input)
        
        # Aggregate predictions from the batch: average is a common strategy
        # This gives a single 108D vector representing the new emotional input impulse
        new_emotion_input_impulse = np.mean(predicted_intensities_batch, axis=0) 

        # Update current emotional state by blending new input with existing state
        # This allows for emotional persistence and smooth transitions
        self.emotional_state = (self.emotional_state * (1 - self.blending_factor) + 
                                new_emotion_input_impulse * self.blending_factor)
        
        # Ensure state is within valid bounds (0 to 1 intensity) after blending
        self.emotional_state = np.clip(self.emotional_state, 0.0, 1.0)
        
        return self.emotional_state.copy()

    # --- Backward Compatibility Wrapper ---
    def process_emotions(self, cognitive_state: np.ndarray) -> np.ndarray:
        """Legacy interface expected by main.py.

        Since the new design maps CAFVE tokens to emotion, if only a cognitive_state
        is provided we derive a lightweight pseudo-token via deterministic projection.
        This avoids mock randomness while preserving structural mapping.
        """
        try:
            if not hasattr(self, '_legacy_proj'):
                rng = np.random.default_rng(seed=123)
                self._legacy_proj = rng.standard_normal((cognitive_state.shape[0], self.input_dim)).astype(np.float32) * (1.0 / np.sqrt(cognitive_state.shape[0]))
            pseudo_token = (cognitive_state.astype(np.float32) @ self._legacy_proj).astype(np.float32)
            # Clamp to reasonable range then sigmoid-like squashing
            pseudo_token = np.tanh(pseudo_token)
            # Use existing pipeline expecting a batch
            return self.process_input([pseudo_token])
        except Exception as e:
            logger.error(f"EmotionCore.process_emotions error: {e}")
            return self.emotional_state.copy()

    def get_emotional_state(self) -> np.ndarray:
        """Returns the current, up-to-date 108-dimensional emotional state vector."""
        # Always apply decay to ensure the returned state is current
        self._apply_temporal_decay() 
        return self.emotional_state.copy()

    def get_persona_emotional_output(self) -> np.ndarray:
        """
        Returns a filtered emotional state for persona impact, ignoring low-value outliers.
        Higher-level modules like the Manifold/Mind will use this filtered output.
        """
        current_state = self.get_emotional_state() # Get up-to-date state
        
        # Create a copy and zero out emotions below the outlier threshold
        filtered_state = current_state.copy()
        filtered_state[filtered_state < self.persona_outlier_threshold] = 0.0
        
        return filtered_state

    def get_emotion_names(self) -> List[str]:
        """Returns the list of 108 emotion names in order."""
        return self.EMOTION_DIMENSIONS

    def get_emotion_index(self, emotion_name: str) -> Optional[int]:
        """Returns the index of an emotion name."""
        return self.EMOTION_NAME_TO_INDEX.get(emotion_name)

                       # self.last_update_times = np.array(state['last_update_times'], dtype=np.float32)
                        
                        #loaded_params = state.get('emotional_mapper_weights', [])
                        
                        #3current_params = self.emotional_mapper.get_trainable_params()
                        
                        #if len(loaded_params) == len(current_params):
                         #   for i, (param_val_list, grad_name_str) in enumerate(loaded_params):
                         #       param_array, _, layer_instance = current_params[i]
#                                param_array[:] = np.array(param_val_list, dtype=np.float32)
#                        else:
 #                           logger.warning("Emotional mapper weights mismatch. Initializing mapper randomly.")
                        
  #                      logger.info(f"EmotionCore state loaded from single document.")
    #                else:
   #                     logger.warning("Loaded EmotionCore state dimensions mismatch. Using defaults.")
     #                   self.emotional_state = np.zeros(self.output_dim, dtype=np.float32)
      #                  self.last_update_times = np.full(self.output_dim, time.perf_counter(), dtype=np.float32)
       #         else:
        #            logger.warning("Emotion state not found in single document. Using defaults.")
         #           self.emotional_state = np.zeros(self.output_dim, dtype=np.float32)
          #          self.last_update_times = np.full(self.output_dim, time.perf_counter(), dtype=np.float32)
#            else:
 #               logger.warning(f"Single document not found. Using defaults.")
  #              self.emotional_state = np.zeros(self.output_dim, dtype=np.float32)
   #             self.last_update_times = np.full(self.output_dim, time.perf_counter(), dtype=np.float32)
    #    except Exception as e:
     #       logger.error(f"Error loading EmotionCore state: {e}. Using defaults.")
    #self.emotional_state = np.zeros(self.output_dim, dtype=np.float32
    #self.last_update_times = np.full(self.output_dim, time.perf_counter(), dtype=np.float32)


# Test block (can be removed in final deployment)
if __name__ == "__main__":
    logger.info("Running EmotionCore test.")

    # Define dimensions for the test
    CAFVE_INPUT_DIM = 80 
    
    # Instantiate EmotionCore
    emotion_core = EmotionCore(input_dim=CAFVE_INPUT_DIM) 
    
    # Simulate an emotionally neutral/low input token (e.g., from CAFVE)
    neutral_token = np.random.rand(CAFVE_INPUT_DIM) * 0.1
    # Simulate a positive input token (e.g., from CAFVE sensing joy/good)
    positive_token_impulse = np.random.rand(CAFVE_INPUT_DIM) * 0.8 + 0.2
    # Simulate a negative input token
    negative_token_impulse = np.random.rand(CAFVE_INPUT_DIM) * 0.8 + 0.2 # Will be interpreted negatively by mapper

    logger.info("Initial emotional state (should be zeros, first 5 dims):")
    logger.info(emotion_core.get_emotional_state()[:5])
    
    # Process some positive input
    logger.info("\nProcessing 5 positive tokens...")
    for _ in range(5):
        emotion_core.process_input([positive_token_impulse])
        time.sleep(0.1) # Simulate time passing per cycle
    
    current_emotional_state = emotion_core.get_emotional_state()
    logger.info("Emotional state after positive input (first 5 dims):")
    logger.info(current_emotional_state[:5])
    
    # Check specific positive/negative emotions by name (if they're above threshold)
    # Note: Emotional mapping is learned, so actual emotions may vary.
    happy_idx = emotion_core.get_emotion_index('Happy')
    sad_idx = emotion_core.get_emotion_index('Sad')
    logger.info(f"Happy intensity: {current_emotional_state[happy_idx]:.4f}")
    logger.info(f"Sad intensity: {current_emotional_state[sad_idx]:.4f}")
    
    # Simulate time passing without input (decay)
    logger.info("\nSimulating 2 seconds of no input (decay)...")
    time.sleep(2.0) 
    emotion_core.process_input([]) # Process with no new input to trigger decay
    
    decayed_emotional_state = emotion_core.get_emotional_state()
    logger.info("Emotional state after decay (first 5 dims):")
    logger.info(decayed_emotional_state[:5])
    logger.info(f"Happy intensity after decay: {decayed_emotional_state[happy_idx]:.4f}")
    logger.info(f"Sad intensity after decay: {decayed_emotional_state[sad_idx]:.4f}")

    # Process some negative input
    logger.info("\nProcessing 5 negative tokens...")
    for _ in range(5):
        emotion_core.process_input([negative_token_impulse])
        time.sleep(0.1)
    
    final_emotional_state = emotion_core.get_emotional_state()
    logger.info("Emotional state after negative input (first 5 dims):")
    logger.info(final_emotional_state[:5])
    logger.info(f"Happy intensity after negative input: {final_emotional_state[happy_idx]:.4f}")
    logger.info(f"Sad intensity after negative input: {final_emotional_state[sad_idx]:.4f}")

    # Get filtered persona output
    persona_output = emotion_core.get_persona_emotional_output()
    logger.info("\nFiltered persona output (non-zero intensities only, showing first 5):")
    logger.info(persona_output[persona_output > 0][:5]) # Show only emotions above threshold
# def load_state():


