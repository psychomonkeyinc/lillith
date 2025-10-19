# --- START OF FILE language.py ---
# language.py

import os
import numpy as np
import logging
import pickle
import time
from typing import Dict, List, Optional, Self, Tuple, Any

# Assume nn.py is available for Sequential, Linear, Sigmoid, ReLU, Tanh
from nn import Sequential, Linear, Sigmoid, ReLU, Tanh

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Language:
    """
    Lillith's Emergent Language System: Builds capacity for abstract communication
    through learned sonic patterns and their internal associations, without LLMs.
    """
    def __init__(self,
                 unified_cognitive_state_dim: int = 256, # Input from Mind.py
                 emotional_state_dim: int = 108,         # Input from Emotion.py
                 cafve_token_dim: int = 80,             # Input from CAFVE (abstract tokens)
                 internal_lang_activity_dim: int = 256, # Dimension of Lillith's internal language representation
                 synth_control_dim: int = 50,           # Expected output dimension for synthesizer control
                 ):
        
        self.unified_cognitive_state_dim = unified_cognitive_state_dim
        self.emotional_state_dim = emotional_state_dim
        self.cafve_token_dim = cafve_token_dim
        self.internal_lang_activity_dim = internal_lang_activity_dim
        self.synth_control_dim = synth_control_dim

        # --- Internal Language Activity Network ---
        # Maps Mind state, Emotion state, and CAFVE tokens into Lillith's
        # internal "language of thought" representation.
        self.internal_lang_input_dim = (self.unified_cognitive_state_dim + 
                                        self.emotional_state_dim + 
                                        self.cafve_token_dim)
        
        self.internal_lang_mapper = Sequential(
            Linear(self.internal_lang_input_dim, 512),
            ReLU(),
            Linear(512, self.internal_lang_activity_dim),
            Tanh() # Internal language activity could be bipolar
        )

        # --- Acoustic Encoder (Expression Generation) ---
        # Translates internal language activity into synthesizer control parameters.
        self.acoustic_encoder = Sequential(
            Linear(self.internal_lang_activity_dim, 128),
            ReLU(),
            Linear(128, self.synth_control_dim),
            Sigmoid() # Synth control parameters often 0-1
        )

        # --- Acoustic Decoder (Interpretation) ---
        # Maps perceived CAFVE tokens (representing external sounds) back into
        # Lillith's internal language activity space or directly to cognitive/emotional shifts.
        self.acoustic_decoder = Sequential(
            Linear(self.cafve_token_dim, 128), # Input is CAFVE tokens
            ReLU(),
            Linear(128, self.internal_lang_activity_dim),
            Tanh() # Output to internal language activity space
        )
        
        # Internal state for current internal language activity
        self._current_internal_lang_activity = np.zeros(self.internal_lang_activity_dim, dtype=np.float32)
        
        logger.info(f"Language System initialized. Internal Language Activity Dim: {self.internal_lang_activity_dim}")

    def process_internal_thought(self,
                                 unified_cognitive_state: np.ndarray,
                                 emotional_state: np.ndarray,
                                 incoming_cafve_token: np.ndarray # Average or most recent CAFVE token
                                 ) -> np.ndarray:
        """
        Maps current cognitive, emotional, and perceptual input into Lillith's
        internal "language of thought" representation.
        """
        if (unified_cognitive_state is None or emotional_state is None or 
            incoming_cafve_token is None):
            logger.warning("Language: Missing input for internal thought processing. Returning current activity.")
            return self._current_internal_lang_activity.copy()

        combined_input = np.concatenate([
            unified_cognitive_state,
            emotional_state,
            incoming_cafve_token
        ]).astype(np.float32)

        if combined_input.shape[0] != self.internal_lang_input_dim:
            logger.error(f"Language: Input dimension mismatch for internal mapper. Expected {self.internal_lang_input_dim}, got {combined_input.shape[0]}. Returning current activity.")
            return self._current_internal_lang_activity.copy()

        # Reshape for NN (batch_size=1)
        combined_input = combined_input.reshape(1, -1)
        
        # Update internal language activity
        self._current_internal_lang_activity = self.internal_lang_mapper.forward(combined_input)[0,:]
        
        logger.debug(f"Language: Internal language activity updated. Norm: {np.linalg.norm(self._current_internal_lang_activity):.4f}")
        return self._current_internal_lang_activity.copy()

    def generate_acoustic_expression(self) -> np.ndarray:
        """
        Translates current internal language activity into synthesizer control parameters.
        This is Lillith's emergent "voice."
        """
        # Ensure input is 2D (batch_size=1)
        input_for_encoder = self._current_internal_lang_activity.reshape(1, -1).astype(np.float32)
        
        synth_params = self.acoustic_encoder.forward(input_for_encoder)[0,:]
        
        logger.debug(f"Language: Generated acoustic expression. Output Dim: {synth_params.shape[0]}")
        return synth_params.astype(np.float32)

    # --- Backward Compatibility Wrapper ---
    def process_language(self, unified_cognitive_state: np.ndarray, emotional_state: np.ndarray) -> np.ndarray:
        """Legacy interface expected by main.py (_make_decisions).

        Derives/updates internal language activity and returns synthesizer control params.
        Avoids mock data; uses deterministic projection if CAFVE token missing.
        """
        try:
            # If we lack a CAFVE token path here, synthesize a stable pseudo-token from cognitive state
            if not hasattr(self, '_legacy_token_proj'):
                rng = np.random.default_rng(seed=321)
                self._legacy_token_proj = rng.standard_normal(
                    (unified_cognitive_state.shape[0], self.cafve_token_dim)
                ).astype(np.float32) * (1.0 / np.sqrt(unified_cognitive_state.shape[0]))

            pseudo_token = (unified_cognitive_state.astype(np.float32) @ self._legacy_token_proj).astype(np.float32)
            pseudo_token = np.tanh(pseudo_token)  # keep bounded

            # Update internal language activity from inputs
            self.process_internal_thought(unified_cognitive_state, emotional_state, pseudo_token)

            # Produce control parameters
            return self.generate_acoustic_expression()
        except Exception as e:
            logger.error(f"Language.process_language error: {e}")
            return np.zeros(self.synth_control_dim, dtype=np.float32)

    def interpret_acoustic_input(self, cafve_token_batch: List[np.ndarray]) -> np.ndarray:
        """
        Interprets a batch of incoming CAFVE tokens (representing external acoustic/visual patterns)
        and maps them to Lillith's internal language activity space.
        """
        if not cafve_token_batch:
            logger.warning("Language: No CAFVE tokens to interpret.")
            return np.zeros(self.internal_lang_activity_dim, dtype=np.float32)

        # Average CAFVE tokens for batch interpretation
        batch_input = np.array(cafve_token_batch, dtype=np.float32)
        avg_cafve_token = np.mean(batch_input, axis=0).astype(np.float32)

        if avg_cafve_token.shape[0] != self.cafve_token_dim:
            logger.error(f"Language: CAFVE token dimension mismatch for decoder. Expected {self.cafve_token_dim}, got {avg_cafve_token.shape[0]}. Returning zeros.")
            return np.zeros(self.internal_lang_activity_dim, dtype=np.float32)

        input_for_decoder = avg_cafve_token.reshape(1, -1)
        
        inferred_lang_activity = self.acoustic_decoder.forward(input_for_decoder)[0,:]
        
        logger.debug(f"Language: Interpreted acoustic input. Inferred Language Activity Norm: {np.linalg.norm(inferred_lang_activity):.4f}")
        return inferred_lang_activity.astype(np.float32)

    def get_networks(self) -> List[None]:
        """Returns a list of all internal neural networks for optimization."""
        return [self.internal_lang_mapper, self.acoustic_encoder, self.acoustic_decoder]

    # Persistence methods (save/load state)
    # def save_state(self, save_path: str):
    #     """Saves the Language module's state to a file."""
    #     try:
    #         state = {
    #             'current_internal_lang_activity': self._current_internal_lang_activity.tolist(),
    #             # Save network weights
    #             'internal_lang_mapper_weights': [(p[0].tolist(), p[1]) for p in self.internal_lang_mapper.get_trainable_params()],
    #             'acoustic_encoder_weights': [(p[0].tolist(), p[1]) for p in self.acoustic_encoder.get_trainable_params()],
    #             'acoustic_decoder_weights': [(p[0].tolist(), p[1]) for p in self.acoustic_decoder.get_trainable_params()]
    #         }
    #         with open(save_path, 'wb') as f:
    #             pickle.dump(state, f)
    #         logger.info(f"Language state saved to {save_path}")
    #     except Exception as e:
    #         logger.error(f"Error saving Language state: {e}")

    # def load_state(self, load_path: str):
    #     """Loads the Language module's state from a file."""
    #     try:
    #         with open(load_path, 'rb') as f:
    #             state = pickle.load(f)
    #         
    #         if np.array(state['current_internal_lang_activity']).shape == self._current_internal_lang_activity.shape:
    #             self._current_internal_lang_activity = np.array(state['current_internal_lang_activity'], dtype=np.float32)
    #             
    #             # Load network weights
    #             loaded_mapper_params = state.get('internal_lang_mapper_weights', [])
    #             loaded_encoder_params = state.get('acoustic_encoder_weights', [])
    #             loaded_decoder_params = state.get('acoustic_decoder_weights', [])

    #             current_mapper_params = self.internal_lang_mapper.get_trainable_params()
    #             current_encoder_params = self.acoustic_encoder.get_trainable_params()
    #             current_decoder_params = self.acoustic_decoder.get_trainable_params()

                # if len(loaded_mapper_params) == len(current_mapper_params):
                #     for i, (param_val_list, grad_name_str) in enumerate(loaded_mapper_params):
                #         param_array, _, layer_instance = current_mapper_params[i] 
                #         param_array[:] = np.array(param_val_list, dtype=np.float32)
                # else: logger.warning("Internal lang mapper weights mismatch. Initializing randomly.")
                # 
                # if len(loaded_encoder_params) == len(current_encoder_params):
                #     for i, (param_val_list, grad_name_str) in enumerate(loaded_encoder_params):
                #         param_array, _, layer_instance = current_encoder_params[i] 
                #         param_array[:] = np.array(param_val_list, dtype=np.float32)
                # else: logger.warning("Acoustic encoder weights mismatch. Initializing randomly.")
                #
                # if len(loaded_decoder_params) == len(current_decoder_params):
                #     for i, (param_val_list, grad_name_str) in enumerate(loaded_decoder_params):
                #         param_array, _, layer_instance = current_decoder_params[i] 
                #         param_array[:] = np.array(param_val_list, dtype=np.float32)
                # else: logger.warning("Acoustic decoder weights mismatch. Initializing randomly.")
                #
                # logger.info(f"Language state loaded from {load_path}")
            # else:
            #     logger.warning("Loaded Language state dimensions mismatch. Initializing to default.")
            #     self._current_internal_lang_activity = np.zeros(self.internal_lang_activity_dim, dtype=np.float32)
            #
        # except FileNotFoundError:
        #     logger.warning(f"Language state file not found at {load_path}. Initializing to default.")
        #     self._current_internal_lang_activity = np.zeros(self.internal_lang_activity_dim, dtype=np.float32)
        # except Exception as e:
        #     logger.error(f"Error loading Language state: {e}. Initializing to default.")
            # self._current_internal_lang_activity = np.zeros(self.internal_lang_activity_dim, dtype=np.float32)

    # ---------------------------------------------------------------------------------
    # EMERGENT PHONEME GENERATION (replaces heuristic / rule-based text parsing)
    # ---------------------------------------------------------------------------------
    # The previous rule / template driven phoneme & prosody helpers have been removed
    # to keep language development fully emergent. Instead we expose a minimal,
    # trainable projection from latent internal language activity -> phoneme logits.
    # No hard-coded text patterns, punctuation heuristics, or semantic keyword lists.

    def init_emergent_phoneme_layer(self, inventory: Optional[List[str]] = None):
        """Initialize a lightweight emergent phoneme inventory.

        Args:
            inventory: Optional custom list of phoneme symbols. If None, a small
                       generic inventory is created. This inventory is NOT tied
                       to any external corpus – it is simply a set of discrete
                       articulatory tokens the system can evolve to use.
        """
        if inventory is None:
            inventory = [
                'AH','EH','IH','OW','UH',    # Basic vowels
                'B','D','G','P','T','K',     # Plosives
                'M','N','L','R',             # Sonorants
                'S','Z','F','V','H',         # Fricatives / breath
                'SH','CH','TH',              # Additional fricatives / affricate / dental
                '_'                          # Blank / pause
            ]
        self.phoneme_inventory = inventory
        inv_dim = len(self.phoneme_inventory)
        rng = np.random.default_rng(seed=777)

        class PhonemeProjectionNet:
            def __init__(self, in_dim: int, out_dim: int, rng_inst):
                self.weights = (rng_inst.standard_normal((in_dim, out_dim)).astype(np.float32) *
                                (1.0 / np.sqrt(in_dim)))
                self.weights_gradient = np.zeros_like(self.weights, dtype=np.float32)
            def forward(self, latent: np.ndarray) -> np.ndarray:
                return latent @ self.weights
            def get_trainable_params(self):
                return [(self.weights, 'weights_gradient', self)]
        self._phoneme_proj_net = PhonemeProjectionNet(self.internal_lang_activity_dim, inv_dim, rng)
        logger.info(f"Emergent phoneme layer initialized. Inventory size={inv_dim}")

    def get_phoneme_projection_network(self):
        return getattr(self, '_phoneme_proj_net', None)

    def _phoneme_softmax(self, logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        z = logits / max(temperature, 1e-6)
        z = z - np.max(z)
        exp_z = np.exp(z)
        probs = exp_z / np.sum(exp_z)
        return probs.astype(np.float32)

    def generate_emergent_phoneme_sequence(self,
                                           max_length: int = 12,
                                           temperature: float = 1.0,
                                           update_internal: bool = True,
                                           deterministic: bool = True) -> List[Dict[str, Any]]:
        """Generate a phoneme sequence purely from latent internal language activity.

        This is intentionally minimal: no text input, no rule templates.
        The inventory is a neutral set of articulatory tokens. Sequence dynamics
        arise from the latent state evolution + simple recurrence using selected
        phoneme embedding columns.
        """
        if not hasattr(self, 'phoneme_inventory'):
            self.init_emergent_phoneme_layer()

        seq = []
        latent = self._current_internal_lang_activity.copy().astype(np.float32)
        proj_net = self.get_phoneme_projection_network()
        for step in range(max_length):
            logits = proj_net.forward(latent)  # (inventory_dim,)
            probs = self._phoneme_softmax(logits, temperature=temperature)
            if deterministic:
                idx = int(np.argmax(probs))
            else:
                idx = int(np.random.choice(len(probs), p=probs))
            phoneme = self.phoneme_inventory[idx]
            confidence = float(probs[idx])
            seq.append({'phoneme': phoneme, 'conf': confidence})
            # Simple latent update: incorporate chosen phoneme column (emergent recurrence)
            latent = np.tanh(latent + self._phoneme_proj[:, idx])
        if update_internal:
            # Blend new latent back into internal language activity (lightly)
            self._current_internal_lang_activity = (
                0.9 * self._current_internal_lang_activity + 0.1 * latent
            ).astype(np.float32)
        return seq

    def entropy_uniform_adapt(self, scale: float = 1e-3):
        """Compute a simple entropy-maximizing (uniform-target) gradient for phoneme projection.
        Loss: KL(U || P) where U is uniform over inventory.
        Gradient logits: p - u ; dW = latent[:,None]*(p-u)[None,:]
        Accumulates into weights_gradient for optimizer step.
        """
        proj_net = self.get_phoneme_projection_network()
        if proj_net is None:
            return
        latent = self._current_internal_lang_activity.astype(np.float32)
        if np.linalg.norm(latent) == 0:
            return
        logits = proj_net.forward(latent)
        probs = self._phoneme_softmax(logits)
        u = np.full_like(probs, 1.0 / probs.shape[0], dtype=np.float32)
        grad_logits = (probs - u)  # shape (V,)
        proj_net.weights_gradient = scale * np.outer(latent, grad_logits).astype(np.float32)
        # Light regularization to prevent drift
        proj_net.weights_gradient += 1e-5 * proj_net.weights

    def get_phoneme_inventory(self) -> List[str]:
        return getattr(self, 'phoneme_inventory', [])

    # Placeholder training hook for future self-supervised adaptation
    def adapt_phoneme_projection(self, feedback: List[Tuple[int, float]], lr: float = 1e-3):
        """Lightweight adaptation of phoneme projection using simple feedback.
        feedback: list of (phoneme_index, reward_signal in [-1,1])
        This is a stand-in for more advanced reinforcement / predictive loss.
        """
        if not hasattr(self, '_phoneme_proj') or not feedback:
            return
        for idx, reward in feedback:
            if 0 <= idx < self._phoneme_proj.shape[1]:
                # Hebbian-like adjustment scaled by reward
                self._phoneme_proj[:, idx] += lr * reward * self._current_internal_lang_activity
        # Prevent uncontrolled growth
        self._phoneme_proj = np.clip(self._phoneme_proj, -3.0, 3.0).astype(np.float32)


# Test block (can be removed in final deployment)
if __name__ == "__main__":
    logger.info("Language module loaded successfully.")
