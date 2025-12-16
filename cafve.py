# cafve.py

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import math # For math.isclose for float comparisons
import pickle # For saving/loading tokenizer state

# Assume nn.py is available for ConsciousnessTokenScorer
# from nn import Sequential, Linear, Sigmoid, ReLU 

# Configure logging (module-level logger only; avoid re-calling basicConfig if main already configured)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sensory Feature Extractor (inlined from sfe.py per user request)
# This class is included here so CAFVE can directly access raw SFE outputs
# without requiring changes to main or new files.
# ---------------------------------------------------------------------------
import threading
import time
from typing import Optional, Tuple

# Optional heavy dependencies used only if available
try:
    import sounddevice as sd  # type: ignore
except Exception:
    sd = None

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


class SensoryFeatureExtractor:
    """
    Handles raw audio and video data, extracting relevant
    features and providing the information to other modules.
    Lightweight inlined version of the project's `sfe.py`.
    """
    def __init__(self,
                 audio_sample_rate: int = 44100,
                 audio_chunk_duration: float = 0.1,
                 video_device_index: int = 0,
                 video_frame_rate: int = 30):

        self.audio_sample_rate = audio_sample_rate
        self.audio_chunk_duration = audio_chunk_duration
        self.audio_chunk_size = int(audio_sample_rate * audio_chunk_duration)

        self.video_device_index = video_device_index
        self.video_frame_rate = video_frame_rate

        # Initialize buffers for storing data.
        self.audio_buffer = np.zeros(self.audio_chunk_size, dtype=np.float32)
        self.video_frame_buffer = np.zeros((480, 640, 3), dtype=np.uint8)

        self.lock = threading.Lock()

        self.running = False
        self.thread = None

        self.latest_frame = None

        logger.info(f"SensoryFeatureExtractor initiated: audio_sr={self.audio_sample_rate}, chunk={self.audio_chunk_size}, video_idx={self.video_device_index}")

    def _capture_audio_loop(self):
        if sd is None:
            logger.warning("sounddevice not available; audio capture disabled")
            return
        try:
            with sd.InputStream(samplerate=self.audio_sample_rate,
                                channels=1,
                                blocksize=self.audio_chunk_size,
                                device=None,
                                dtype='float32',
                                callback=self._audio_callback):
                logger.info("Audio Loop Thread: Listening for audio data...")
                while self.running:
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Audio Capture Exception {e}")

    def _capture_video_loop(self):
        if cv2 is None:
            logger.warning("opencv (cv2) not available; video capture disabled")
            return
        cap = cv2.VideoCapture(self.video_device_index)
        if not cap.isOpened():
            logger.warning(f"FAILED TO OPEN VIDEO device index {self.video_device_index}")
            return
        while self.running:
            ret, frame = cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame.copy()
            else:
                time.sleep(0.01)
        cap.release()

    def start(self):
        self.running = True
        self.audio_thread = threading.Thread(target=self._capture_audio_loop, daemon=True)
        self.video_thread = threading.Thread(target=self._capture_video_loop, daemon=True)
        self.audio_thread.start()
        self.video_thread.start()
        logger.info("SensoryFeatureExtractor started")

    def stop(self):
        if self.running:
            self.running = False
            if hasattr(self, 'audio_thread') and self.audio_thread is not None:
                self.audio_thread.join(timeout=2.0)
            if hasattr(self, 'video_thread') and self.video_thread is not None:
                self.video_thread.join(timeout=2.0)
            logger.info("SensoryFeatureExtractor stopped")

    def _audio_callback(self, indata: np.ndarray, frames, time_info, status):
        if status:
            logger.warning(f"AudioIn Warning: {status}")
        with self.lock:
            try:
                self.audio_buffer = indata.copy()
            except Exception:
                pass

    def get_features(self, audio_chunk: Optional[np.ndarray], video_frame: Optional[np.ndarray]) -> np.ndarray:
        """
        Produces a lightweight feature vector (default 55D) from audio and video inputs.
        This is intentionally simple and matches the expectations in `cafve`.
        """
        if video_frame is not None:
            with self.lock:
                # Generate deterministic pseudo-features for compatibility
                # Use simple statistics for stable inputs to downstream modules
                audio_chunk = audio_chunk if audio_chunk is not None else self.audio_buffer
                loudness = float(np.mean(np.abs(audio_chunk))) if audio_chunk is not None else 0.0
                pitch_est = 0.0
                motion = 0.0
                try:
                    if video_frame is not None:
                        # compute simple motion proxy: mean absolute difference from zero
                        motion = float(np.mean(np.abs(video_frame.astype(np.float32)))) / 255.0
                except Exception:
                    motion = 0.0

                # Build a reproducible 55-d vector using repeated copies/slots of basic stats
                vec = np.zeros(55, dtype=np.float32)
                vec[0] = loudness
                vec[1] = pitch_est
                vec[40] = motion
                # Fill remaining with low-variance deterministic values derived from inputs
                idx = 2
                for s in [loudness, pitch_est, motion]:
                    for _ in range(5):
                        if idx < 55:
                            vec[idx] = float(s)
                            idx += 1
                # Fill remainder with small deterministic sequence for stability
                for i in range(idx, 55):
                    vec[i] = float((i % 7) * 0.01)

                return vec
        else:
            return np.zeros(55, dtype=np.float32)

# End inlined SFE

# ============================================================================
# ACE™ TOKENIZER DATA STRUCTURES (ADAPTED FOR FEATURE VECTORS)
# ============================================================================

@dataclass
class TokenMetadata:
    """Metadata for consciousness-aware tokens, now derived from feature vectors."""
    token_id: int
    frequency: int
    semantic_cluster: int
    emotional_weight: float        # Derived from feature vector properties
    consciousness_relevance: float # Derived from feature vector properties
    feminine_linguistic_score: float # Derived from feature vector properties
    creation_timestamp: datetime
    last_used: datetime
    feature_centroid: np.ndarray   # Centroid of the Percept feature vectors that form this token

@dataclass
class ConsciousnessContext:
    """Context for consciousness-aware tokenization, derived from SFE features."""
    # These will be directly from the SFE module's aggregated features
    # Example fields based on SFE's output, simplified for context consumption by scorer
    loudness_avg: float
    pitch_avg: float
    motion_avg: float
    emo_valence_avg: float
    comm_intent_avg: float
    agency_detect_avg: float
    
    # Future dynamic traits or higher-level interpretations
    # personality_traits: Dict[str, float] 
    # semantic_focus: Dict[str, float] 
    # user_interaction_type: str 

@dataclass
class TokenizationResult:
    """Result of ACE™ tokenization, now with feature-vector-based tokens."""
    token_ids: List[int]
    # No raw 'tokens' list as they are abstract feature vectors, use token_ids for lookup
    attention_weights: List[float] # Future dynamic attention support
    semantic_clusters: List[int]
    emotional_scores: List[float]
    consciousness_relevance: List[float]
    metadata: Dict[str, Any]
    original_sfe_sequence: List[np.ndarray] # Sequence of the original SFE feature vectors (e.g., 55D)
    learned_abstract_tokens: List[np.ndarray] # Sequence of the actual learned abstract token vectors (e.g., 80D)

# ============================================================================
# I. CONSCIOUSNESS-AWARE FEATURE VECTOR ENCODER (CAFVE)
# ============================================================================

class ConsciousnessAwareFeatureVectorEncoder:
    """
    CAFVE: Adapts BPE logic to operate on sequences of numerical feature vectors.
    Considers feature similarity, evolution, and consciousness relevance for merging.
    """
    
    def __init__(self, vocab_size: int = 50000, consciousness_weight: float = 0.3, 
                 sfe_feature_dim: int = 55, token_output_dim: int = 80): # <-- CORRECTED 'feature_dim' to 'sfe_feature_dim'
        
        self.vocab_size = vocab_size
        self.consciousness_weight = np.float32(consciousness_weight)
        self.sfe_feature_dim = sfe_feature_dim # Expected input dimension from SFE module (~55D)
        self.token_output_dim = token_output_dim # Output dimension for abstract tokens (~80D for SOM input)
        
        # Core vocabulary components
        self.feature_vector_frequencies = Counter() # Counts occurrences of unique SFE feature vectors
        self.pair_frequencies = Counter() # Counts occurrences of consecutive SFE feature vector pairs
        self.token_metadata = {} # Stores metadata for learned abstract tokens
        self.merges = [] # Stores ordered list of merged (sfe_vec_A, sfe_vec_B) -> new_token_vector

        # --- Consciousness-specific components (adapted to feature patterns) ---
        self.semantic_clusters = {} # Clusters of learned abstract feature tokens
        # Feminine patterns: now represented as learned thresholds/ranges in feature space
        # Initial values; can be learned/tuned over time.
        self.feminine_feature_patterns = {
            'empathy_cues': {'VocalTone_range': (0.5, 1.0), 'FaceExp_Joy_range': (0.5, 1.0), 'Gaze_Direct_threshold': 0.8}, 
            'nurturing_cues': {'Arousal_range': (0.0, 0.4), 'BodyPose_Open_threshold': 0.7, 'VocalTone_range': (0.4, 0.8)},
            # ... additional patterns for collaborative, expressive, emotional descriptors
        }
        
        # Special tokens: now abstract feature vectors themselves
        # These are initial abstract concepts represented as vectors in the output token space (~80D)
        self.special_tokens_vectors = {
            '<PAD>': np.zeros(self.token_output_dim, dtype=np.float32),
            '<UNK>': np.random.uniform(-0.1, 0.1, self.token_output_dim).astype(np.float32), 
            '<BOS>': np.full(self.token_output_dim, 0.1, dtype=np.float32),
            '<EOS>': np.full(self.token_output_dim, -0.1, dtype=np.float32),
            '<EMOTION>': np.random.uniform(0.1, 0.3, self.token_output_dim).astype(np.float32),
            '<THOUGHT>': np.random.uniform(0.3, 0.5, self.token_output_dim).astype(np.float32),
            '<INTENTION>': np.random.uniform(0.5, 0.7, self.token_output_dim).astype(np.float32),
            '<MEMORY>': np.random.uniform(0.7, 0.9, self.token_output_dim).astype(np.float32),
            '<EMPATHY>': np.random.uniform(0.0, 0.2, self.token_output_dim).astype(np.float32),
            '<CURIOSITY>': np.random.uniform(0.2, 0.4, self.token_output_dim).astype(np.float32),
            '<AFFECTION>': np.random.uniform(0.4, 0.6, self.token_output_dim).astype(np.float32),
            '<VOCAL_ATTEMPT>': np.random.uniform(0.6, 0.8, self.token_output_dim).astype(np.float32),
            '<CONSCIOUSNESS>': np.random.uniform(0.8, 1.0, self.token_output_dim).astype(np.float32)
        }
        
        # Mapping from integer ID to abstract feature vector (the "token")
        # Ensure these are distinct. Use a very small tolerance for comparison when adding.
        self.id_to_token_vector = {}
        for i, vec in enumerate(self.special_tokens_vectors.values()):
            self.id_to_token_vector[i] = vec
        
        # Mapping from abstract feature vector to integer ID (using tuple for hashable key)
        self.token_vector_to_id = {tuple(vec.tolist()): i for i, vec in self.id_to_token_vector.items()}
        
        self.next_token_id = len(self.special_tokens_vectors) # Next available ID for learned tokens
        
        logger.info(f"CAFVE initialized: SFE_dim={self.sfe_feature_dim}, Token_out_dim={self.token_output_dim}")

    def _vector_to_id(self, vector: np.ndarray) -> int:
        """Helper to get ID from feature vector, handling new unique vectors."""
        vec_tuple = tuple(vector.tolist())
        # Check if vector already exists within a small tolerance for float equality
        for stored_vec_tuple, stored_id in self.token_vector_to_id.items():
            if np.allclose(np.array(stored_vec_tuple), vector):
                return stored_id
        
        # If not found, assign new ID
        self.token_vector_to_id[vec_tuple] = self.next_token_id
        self.id_to_token_vector[self.next_token_id] = vector
        self.next_token_id += 1
        return self.token_vector_to_id[vec_tuple]

    def _id_to_vector(self, token_id: int) -> np.ndarray:
        """Helper to get feature vector from ID."""
        return self.id_to_token_vector.get(token_id, self.special_tokens_vectors['<UNK>'])

    def _get_pair_frequencies(self, sfe_vector_sequence: List[np.ndarray]) -> Counter:
        """Counts frequencies of all adjacent feature vector pairs."""
        pair_counts = Counter()
        for i in range(len(sfe_vector_sequence) - 1):
            # Using tuple of list for hashable key, as np.ndarray is not hashable
            pair = (tuple(sfe_vector_sequence[i].tolist()), tuple(sfe_vector_sequence[i+1].tolist()))
            pair_counts[pair] += 1
        return pair_counts

    def _calculate_pair_consciousness_score(self, pair: Tuple[np.ndarray, np.ndarray], context_features: Dict[str, float]) -> float:
        """
        Calculates consciousness relevance score for a feature vector pair.
        This is a heuristic, will be refined by ConsciousnessTokenScorer NN.
        """
        vec1 = np.array(pair[0], dtype=np.float32)
        vec2 = np.array(pair[1], dtype=np.float32)
        
        # Feature Similarity (cosine similarity)
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
        similarity = (similarity + 1) / 2 # Normalize to 0-1 range from -1 to 1

        # Feature Evolution (magnitude of change)
        change_magnitude = np.linalg.norm(vec2 - vec1)
        evolution_score = 1.0 - np.tanh(change_magnitude / self.sfe_feature_dim) # Normalized by input dim

        # Aggregate inferred consciousness features from the context (overall SFE segment)
        # These indices map to specific features in the 55D SFE vector
        # This mapping needs to be consistent with sfe.py's output order
        # Assuming order from previous turn: Loudness, PitchFluct, etc., up to CommIntent
        
        # Mapping context_features keys to SFE vector indices (must be kept consistent with sfe.py)
        # For simplicity, let's use example indices from a hypothetical 55D SFE vector for context
        # (This would be more robust with named indices in SFE output)
        
        # Using context_features that are explicitly passed (Context from SFE aggregated features)
        percept_agency = context_features.get('AgencyDetect', 0.0)
        percept_emo_valence = context_features.get('EmoValence', 0.0)
        percept_comm_intent = context_features.get('CommIntent', 0.0)
        percept_arousal = context_features.get('Arousal', 0.0)
        percept_vocal_tone = context_features.get('VocalTone', 0.0)

        # Heuristic for consciousness relevance of the pair
        # Emphasize intentionality, emotional shifts, and coherence
        base_relevance = (percept_agency * 0.4 + 
                          (percept_emo_valence + 1)/2 * 0.3 + 
                          percept_comm_intent * 0.3)
        
        # Combine: similarity for coherence, evolution for meaningful change, base_relevance for high-level context
        combined_score = (similarity * 0.4 + evolution_score * 0.3 + base_relevance * 0.3)
        
        # Feminine feature pattern bonus (simplified check based on ranges and context features)
        feminine_bonus = 0.0
        # Example: check if features like VocalTone are within 'empathy_cues' ranges
        # This relies on accurate range lookups from self.feminine_feature_patterns
        if (self.feminine_feature_patterns['empathy_cues']['VocalTone_range'][0] <= percept_vocal_tone <= self.feminine_feature_patterns['empathy_cues']['VocalTone_range'][1] and
            self.feminine_feature_patterns['empathy_cues']['Arousal_range'][0] <= percept_arousal <= self.feminine_feature_patterns['empathy_cues']['Arousal_range'][1]):
            feminine_bonus += 0.1 # Simplified, actual check would involve multiple features/conditions

        final_score = np.clip(combined_score + feminine_bonus, 0.0, 1.0)
        return final_score.astype(np.float32)

    def _select_best_pair(self, pair_counts: Counter, pair_consciousness_scores: Dict[Tuple[Any, Any], float]) -> Optional[Tuple[Tuple, Tuple]]:
        """
        Selects the best pair to merge based on frequency and consciousness score.
        Input pairs are tuples of feature vectors (as tuples themselves for hashability).
        """
        best_pair_tuple = None
        best_score = -1.0
        
        # Consider top N most frequent pairs for efficiency
        for pair_tup, frequency in pair_counts.most_common(500): # Increased consideration range
            if frequency < 2: # Skip very rare pairs
                continue
            
            # Average consciousness score for this pair
            avg_consciousness_score = pair_consciousness_scores.get(pair_tup, 0.0)
            
            # Combine frequency and consciousness score (using self.consciousness_weight)
            frequency_score = np.float32(frequency) / np.float32(max(pair_counts.values()) + 1e-8) # Normalize frequency
            combined_score = (1.0 - self.consciousness_weight) * frequency_score + \
                             self.consciousness_weight * avg_consciousness_score
            
            if combined_score > best_score:
                best_score = combined_score
                best_pair_tuple = pair_tup 

        return best_pair_tuple

    def _create_merged_vector(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """
        Creates a new abstract feature vector (~80D) from two merged ~55D SFE vectors.
        This represents the core of the abstract token generation.
        """
        # Learned projection/compression for token generation.
        # Uses a small, internal NumPy-based neural network (from nn.py)
        # to perform this specific transformation/compression.
        # Weights are learned during the vocabulary building process.

        # Implementation: concatenation + dimensionality reduction (PCA-like or learned linear projection).
        # Targeting 80D output from two 55D inputs (110D concatenated) requires reduction.
        
        combined = np.concatenate([vec1, vec2]).astype(np.float32)
        
        # --- Learned Dimensionality Reduction ---
        # Instance of Sequential(Linear(110, 80), Sigmoid/Tanh) from nn.py
        # Weights are part of the CAFVE's learnable parameters and saved/loaded.
        
        # Simple linear projection/truncation:
        # Create a random projection matrix if not already exists (should be part of learned parameters)
        if not hasattr(self, '_projection_matrix') or self._projection_matrix.shape != (combined.shape[0], self.token_output_dim):
            self._projection_matrix = np.random.randn(combined.shape[0], self.token_output_dim).astype(np.float32) * 0.01

        # Apply a simple linear projection (non-learned yet)
        abstract_token_vector = np.dot(combined, self._projection_matrix)
        
        # Apply a non-linearity to keep values bounded, simulating activation
        abstract_token_vector = np.tanh(abstract_token_vector) # Or sigmoid, ReLU
        
        return abstract_token_vector.astype(np.float32)

    def build_vocabulary(self, sfe_vector_sequences: List[List[np.ndarray]], 
                         consciousness_contexts: Optional[List[ConsciousnessContext]] = None):
        """
        Builds the consciousness-aware vocabulary of abstract feature tokens.
        `sfe_vector_sequences` are lists of lists, where each inner list is a sequence
        of ~55D SFE feature vectors.
        """
        logger.info("Building ACE™ Consciousness-Aware Feature Vector Vocabulary...")
        
        # Initialize vocabulary with individual SFE feature vectors (as base "tokens")
        # These individual SFE vectors will be mapped into the token_output_dim space
        initial_tokens_vectors_mapped = []
        for seq in sfe_vector_sequences:
            for vec in seq:
                mapped_vec = self._map_sfe_to_token_space(vec) # Map 55D to 80D for base tokens
                if tuple(mapped_vec.tolist()) not in self.token_vector_to_id:
                    self._vector_to_id(mapped_vec) 
                initial_tokens_vectors_mapped.append(mapped_vec)

        logger.info(f"Initial vocabulary size: {len(self.id_to_token_vector)}")
        
        # Iteratively merge pairs based on consciousness-aware scoring
        current_sequences_of_token_ids = []
        for seq in sfe_vector_sequences:
            current_sequences_of_token_ids.append([self._vector_to_id(self._map_sfe_to_token_space(vec)) for vec in seq])
        
        # For scoring: aggregate contexts from original SFE sequences
        # This maps the 55D SFE features to simpler context_features for the scorer
        aggregated_contexts = [self._aggregate_sfe_features_for_context(seq) for seq in sfe_vector_sequences]

        iteration_count = 0
        while self.next_token_id < self.vocab_size:
            iteration_count += 1
            pair_counts = Counter()
            pair_consciousness_scores = defaultdict(float) 
            pair_occurrence_counts = Counter()
            
            # Iterate through current sequences of token IDs to find pairs
            for seq_idx, token_ids_seq in enumerate(current_sequences_of_token_ids):
                context_for_scoring = aggregated_contexts[seq_idx]
                
                for i in range(len(token_ids_seq) - 1):
                    # Get the actual token vectors for the pair of IDs
                    vec_id1 = token_ids_seq[i]
                    vec_id2 = token_ids_seq[i+1]
                    vec1 = self._id_to_vector(vec_id1)
                    vec2 = self._id_to_vector(vec_id2)

                    # Using tuple of list for hashable key, representing the actual feature vectors
                    pair_tuple = (tuple(vec1.tolist()), tuple(vec2.tolist()))
                    
                    pair_counts[pair_tuple] += 1
                    
                    # Calculate consciousness score for this specific pair
                    score = self._calculate_pair_consciousness_score((vec1, vec2), context_for_scoring)

                    pair_consciousness_scores[pair_tuple] += score
                    pair_occurrence_counts[pair_tuple] += 1
            
            # Calculate average consciousness score for each pair
            for pair_tuple in pair_consciousness_scores:
                if pair_occurrence_counts[pair_tuple] > 0:
                    pair_consciousness_scores[pair_tuple] /= pair_occurrence_counts[pair_tuple]

            if not pair_counts:
                logger.info("No more pairs to merge. Vocabulary building stopped.")
                break
            
            best_pair_tuple = self._select_best_pair(pair_counts, pair_consciousness_scores)
            
            if best_pair_tuple is None:
                logger.info("No suitable pair found for merge. Vocabulary building stopped.")
                break
            
            # Retrieve actual vectors from tuple for merging
            best_vec1 = np.array(best_pair_tuple[0], dtype=np.float32)
            best_vec2 = np.array(best_pair_tuple[1], dtype=np.float32)

            new_abstract_token_vector = self._create_merged_vector(best_vec1, best_vec2)
            new_abstract_token_id = self._vector_to_id(new_abstract_token_vector) # This also stores the vector
            
            self.merges.append((best_pair_tuple, new_abstract_token_vector)) # Store the original vectors and the new token

            # Update all sequences with the new merged token ID
            for seq_idx in range(len(current_sequences_of_token_ids)):
                temp_seq = []
                i = 0
                while i < len(current_sequences_of_token_ids[seq_idx]):
                    current_id1 = current_sequences_of_token_ids[seq_idx][i]
                    current_vec1 = self._id_to_vector(current_id1)
                    
                    if i + 1 < len(current_sequences_of_token_ids[seq_idx]):
                        current_id2 = current_sequences_of_token_ids[seq_idx][i+1]
                        current_vec2 = self._id_to_vector(current_id2)
                        
                        # Compare vectors using np.allclose due to float precision
                        if np.allclose(current_vec1, best_vec1) and np.allclose(current_vec2, best_vec2):
                            temp_seq.append(new_abstract_token_id)
                            i += 2 # Skip the next vector as it's merged
                            continue
                    
                    temp_seq.append(current_id1)
                    i += 1
                current_sequences_of_token_ids[seq_idx] = temp_seq

            # Create metadata for new abstract token
            avg_consciousness_score_for_merge = pair_consciousness_scores.get(best_pair_tuple, 0.0)
            self.token_metadata[new_abstract_token_id] = TokenMetadata(
                token_id=new_abstract_token_id,
                frequency=pair_counts[best_pair_tuple],
                semantic_cluster=0,  # Will be assigned later
                emotional_weight=avg_consciousness_score_for_merge,
                consciousness_relevance=avg_consciousness_score_for_merge,
                feminine_linguistic_score=self._calculate_feminine_score_from_vector(new_abstract_token_vector),
                creation_timestamp=datetime.now(),
                last_used=datetime.now(),
                feature_centroid=new_abstract_token_vector.copy() # Store the vector itself
            )
            
            if iteration_count % 100 == 0:
                logger.info(f"Iteration {iteration_count}: Vocab size: {len(self.id_to_token_vector)}, Latest merge ID: {new_abstract_token_id}")
        
        logger.info(f"Final vocabulary size: {len(self.id_to_token_vector)}")
        logger.info(f"Total merges: {len(self.merges)}")
        
        self._build_semantic_clusters() # Cluster the abstract feature tokens
        logger.info("ACE™ feature vector vocabulary building complete!")

    def _map_sfe_to_token_space(self, sfe_vector: np.ndarray) -> np.ndarray:
        """
        Maps a ~55D SFE vector to the ~80D token space.
        This is used for initial base tokens and requires a learned transformation.
        """
        # This would be a small NN (Sequential, Linear) from nn.py
        # trained to project SFE's 55D features into the 80D token space.
        # For initial implementation: a simple linear projection + tanh
        
        if not hasattr(self, '_sfe_to_token_projection_matrix') or self._sfe_to_token_projection_matrix.shape != (sfe_vector.shape[0], self.token_output_dim):
            self._sfe_to_token_projection_matrix = np.random.randn(sfe_vector.shape[0], self.token_output_dim).astype(np.float32) * 0.01

        mapped_vector = np.dot(sfe_vector, self._sfe_to_token_projection_matrix)
        mapped_vector = np.tanh(mapped_vector) # Non-linearity
        return mapped_vector.astype(np.float32)

    def _aggregate_sfe_features_for_context(self, sfe_sequence: List[np.ndarray]) -> ConsciousnessContext:
        """
        Aggregates features from a sequence of SFE vectors for context scoring.
        Returns a ConsciousnessContext object.
        """
        if not sfe_sequence:
            return ConsciousnessContext(loudness_avg=0.0, pitch_avg=0.0, motion_avg=0.0, 
                                        emo_valence_avg=0.0, comm_intent_avg=0.0, agency_detect_avg=0.0)
        
        # Assume sfe_sequence elements are ~55D and we know their indices
        # Example mapping of names to SFE vector indices (must be consistent with sfe.py):
        # Audio Dynamics: [0]Loudness, [1]PitchFluct, [2]Rhythmicity, [3]VocalTone, [4]Arousal
        # Visual Dynamics: [30]FaceExp (vector), [33]Gaze (vector), [35]BodyPose (vector), [40]Motion, [41]SceneChg
        # Inferred: [52]AgencyDetect, [53]EmoValence, [54]CommIntent

        # Aggregate by averaging relevant scalar features
        loudness_sum = 0.0; pitch_sum = 0.0; motion_sum = 0.0; emo_val_sum = 0.0; comm_int_sum = 0.0; agency_sum = 0.0
        
        for vec in sfe_sequence:
            if vec.shape[0] >= 55: # Ensure vector is correct size
                loudness_sum += vec[0]
                pitch_sum += vec[1]
                # Motion is at index 40, assuming previous sfe.py breakdown sum to 55D
                motion_sum += vec[40] 
                emo_val_sum += vec[53] 
                comm_int_sum += vec[54]
                agency_sum += vec[52]
        
        count = len(sfe_sequence)
        
        return ConsciousnessContext(
            loudness_avg=loudness_sum/count, 
            pitch_avg=pitch_sum/count, 
            motion_avg=motion_sum/count,
            emo_valence_avg=emo_val_sum/count,
            comm_intent_avg=comm_int_sum/count,
            agency_detect_avg=agency_sum/count
        )

    def _calculate_feminine_score_from_vector(self, feature_vector: np.ndarray) -> float:
        """
        Calculates feminine linguistic score from an abstract feature vector (~80D).
        This will check if patterns in the vector align with feminine feature patterns.
        """
        score = 0.0
        # This needs a sophisticated learned function, mapping specific feature ranges/combinations
        # within the 80D abstract token vector to a "feminine score".
        
        # Heuristic based on expected influence on token features:
        # Assuming abstract token vector has embedded properties related to:
        # idx 0-4: mapped from audio dynamics
        # idx 5-14: mapped from audio context
        # idx 15-29: mapped from voiceprint
        # idx 30-32: mapped from FaceExp
        # idx 33-34: mapped from Gaze
        # idx 35-39: mapped from BodyPose
        # idx 40-41: mapped from Motion/SceneChg
        # idx 42-51: mapped from Visual Context
        # idx 52-54: mapped from Inferred Intentionality
        
        # Example heuristic: if abstract token emphasizes warmth (from vocal tone) and openness (from body pose)
        # Assuming mapped vocal tone is ~token_vector[3], openness is ~token_vector[35+BodyPose_Open_idx]
        
        # Heuristic calculation:
        if feature_vector[3] > 0.5 and feature_vector[35] > 0.5: # Example: mapped VocalTone high, BodyPose component high for openness
            score += 0.5
        if feature_vector[30] > 0.7: # Example: mapped FaceExp_Joy component high
            score += 0.3
        
        return np.clip(score, 0.0, 1.0) # Ensure 0-1 range

    def _build_semantic_clusters(self):
        """
        Builds semantic clusters for abstract feature tokens using a simple heuristic.
        In a real system, this would be a more sophisticated clustering algorithm (e.g., k-means)
        on the feature_centroid of the tokens.
        """
        logger.info("Building semantic clusters for feature tokens...")
        
        clusters = {
            'emotional_dominant': [], 
            'intentional_dominant': [], 
            'calm_stable': [], 
            'dynamic_change': [], 
            'self_awareness': [], # New cluster for tokens related to internal state
            'other_awareness': [], # New cluster for tokens related to external agents
            'unknown_misc': []
        }
        
        for token_id, metadata in self.token_metadata.items():
            centroid = metadata.feature_centroid # The ~80D abstract token vector
            
            # This logic assumes the 80D token has learned to map SFE features in a somewhat
            # consistent way, which is a big assumption for this heuristic.
            # The indices below are conceptual mappings for where SFE features influence the token.
            # Example: EmoValence (SFE index ~63), Arousal (SFE index ~4), Agency (SFE index ~62), CommIntent (SFE index ~64), Motion (SFE index ~49)

            # This is a VERY rough heuristic for initial clustering.
            # It checks for dominant features in the abstract token vector.
            if np.abs(centroid[63]) > 0.7 or centroid[4] > 0.7: # High EmoValence or Arousal
                clusters['emotional_dominant'].append(token_id)
            elif centroid[62] > 0.7 or centroid[64] > 0.7: # High AgencyDetect or CommIntent
                clusters['intentional_dominant'].append(token_id)
            elif centroid[4] < 0.3 and centroid[49] < 0.3: # Low Arousal and Motion
                clusters['calm_stable'].append(token_id)
            elif centroid[49] > 0.7: # High Motion or SceneChg
                clusters['dynamic_change'].append(token_id)
            elif np.mean(np.abs(centroid[0:30])) < 0.2: # If audio features are very low (internal focus?)
                 clusters['self_awareness_proxy'].append(token_id)
            elif np.mean(np.abs(centroid[30:60])) > 0.6: # If video features are very high (external focus?)
                 clusters['other_awareness_proxy'].append(token_id)
            else:
                clusters['unknown_misc'].append(token_id) 
        
        # Assign cluster IDs and update metadata
        cluster_id = 0
        for cluster_name, token_ids in clusters.items():
            for token_id in token_ids:
                if token_id in self.token_metadata:
                    self.token_metadata[token_id].semantic_cluster = cluster_id
            cluster_id += 1
        
        self.semantic_clusters = clusters
        logger.info(f"Created {len(clusters)} semantic clusters for feature tokens.")
                
    def tokenize(self, sfe_vector_sequence: List[np.ndarray], 
                 consciousness_context: Optional[ConsciousnessContext] = None) -> TokenizationResult:
        """
        Tokenizes a sequence of SFE feature vectors into ACE™ abstract feature tokens.
        Returns a TokenizationResult with token IDs and associated metadata.
        Deterministic, procedural implementation with a simple consciousness-aware
        greedy pair-merge strategy (no learned weights required at runtime).
        """
        # Ensure result fields are always present (avoid partially filled returns)
        if not sfe_vector_sequence:
            return TokenizationResult(
                token_ids=[],
                attention_weights=[],
                semantic_clusters=[],
                emotional_scores=[],
                consciousness_relevance=[],
                metadata={},
                original_sfe_sequence=[],
                learned_abstract_tokens=[]
            )

        # Aggregate or use provided context to derive scoring features
        if consciousness_context is None:
            aggregated_context = self._aggregate_sfe_features_for_context(sfe_vector_sequence)
        else:
            aggregated_context = consciousness_context

        # Map aggregated context to the simple dict expected by _calculate_pair_consciousness_score
        context_features = {
            'AgencyDetect': getattr(aggregated_context, 'agency_detect_avg', 0.0),
            'EmoValence': getattr(aggregated_context, 'emo_valence_avg', 0.0),
            'CommIntent': getattr(aggregated_context, 'comm_intent_avg', 0.0),
            'Arousal': getattr(aggregated_context, 'motion_avg', 0.0),
            'VocalTone': getattr(aggregated_context, 'pitch_avg', 0.0)
        }

        # 1) Map all SFE vectors into the token space (~80D) and assign base token IDs
        mapped_vectors: List[np.ndarray] = []
        base_token_ids: List[int] = []
        for sfe in sfe_vector_sequence:
            sfe_arr = np.asarray(sfe, dtype=np.float32)
            if sfe_arr.size != self.sfe_feature_dim:
                # normalize/resize to expected SFE dimension for safety
                sfe_arr = np.resize(sfe_arr, (self.sfe_feature_dim,))
            mapped = self._map_sfe_to_token_space(sfe_arr)
            mapped_vectors.append(mapped)
            tid = self._vector_to_id(mapped)
            base_token_ids.append(tid)

        # 2) Greedy non-overlapping consciousness-aware merging of adjacent token pairs
        merged_token_ids: List[int] = []
        created_token_ids: List[int] = []
        i = 0
        # Merge threshold can be tuned; moderate default to prefer meaningful merges
        merge_threshold = getattr(self, 'merge_threshold', 0.65)

        while i < len(mapped_vectors):
            if i + 1 < len(mapped_vectors):
                vec1 = mapped_vectors[i]
                vec2 = mapped_vectors[i+1]
                pair_score = float(self._calculate_pair_consciousness_score((vec1, vec2), context_features))
                # If pair is considered strongly conscious/meaningful, merge them
                if pair_score >= merge_threshold:
                    new_vec = self._create_merged_vector(vec1, vec2)
                    new_id = self._vector_to_id(new_vec)
                    merged_token_ids.append(new_id)
                    created_token_ids.append(new_id)
                    # Ensure metadata exists for newly created merged token
                    if new_id not in self.token_metadata:
                        self.token_metadata[new_id] = TokenMetadata(
                            token_id=new_id,
                            frequency=1,
                            semantic_cluster=-1,
                            emotional_weight=pair_score,
                            consciousness_relevance=pair_score,
                            feminine_linguistic_score=self._calculate_feminine_score_from_vector(new_vec),
                            creation_timestamp=datetime.now(),
                            last_used=datetime.now(),
                            feature_centroid=new_vec.copy()
                        )
                    i += 2
                    continue
            # Otherwise, keep current token id
            merged_token_ids.append(base_token_ids[i])
            i += 1

        # 3) Build outputs: attention, clusters, emotional/consciousness scores, learned vectors
        attention_weights = [1.0 for _ in merged_token_ids]
        # Normalize attention to sum to 1 for stability (if non-empty)
        total_att = sum(attention_weights)
        if total_att > 0:
            attention_weights = [float(a) / float(total_att) for a in attention_weights]

        semantic_clusters = []
        emotional_scores = []
        consciousness_scores = []
        learned_token_vectors: List[np.ndarray] = []

        for tid in merged_token_ids:
            # token vector
            vec = self._id_to_vector(tid)
            learned_token_vectors.append(vec)
            # metadata lookups (with sensible defaults)
            meta = self.token_metadata.get(tid, None)
            semantic_clusters.append(meta.semantic_cluster if meta is not None else -1)
            emotional_scores.append(float(meta.emotional_weight) if meta is not None else 0.0)
            consciousness_scores.append(float(meta.consciousness_relevance) if meta is not None else 0.0)
            # update last_used timestamp if metadata exists
            if meta is not None:
                meta.last_used = datetime.now()

        # Prepare result metadata summary
        result_metadata = {
            'original_length': len(sfe_vector_sequence),
            'mapped_length': len(mapped_vectors),
            'final_token_count': len(merged_token_ids),
            'created_token_ids': created_token_ids,
            'merge_threshold': merge_threshold,
            'context_features_snapshot': context_features
        }

        return TokenizationResult(
            token_ids=merged_token_ids,
            attention_weights=attention_weights,
            semantic_clusters=semantic_clusters,
            emotional_scores=emotional_scores,
            consciousness_relevance=consciousness_scores,
            metadata=result_metadata,
            original_sfe_sequence=[np.asarray(v, dtype=np.float32) for v in sfe_vector_sequence],
            learned_abstract_tokens=learned_token_vectors
        )