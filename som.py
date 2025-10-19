#Line1-1505 SoM JustinJ Optimizer Neural Network
#line 1505 starts the biology 

import numpy as np
import time
import logging
from pathlib import Path
from typing import Optional, Tuple 

# Single consolidated state file for all modules

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dimension constants for import compatibility
SOM_ACTIVATION_DIM = 64  # Default input dimension for SOM (fixed, no dynamic growth)
SOM_BMU_COORD_DIM = 2    # X,Y coordinates of best matching unit

class SelfOrganizingMap:
    """
    A Complex Adaptive System (CAS) SOM for foundational cognitive processing.
    Designed with parameters geared towards productive fragility and organic learning.
    Supports dynamic growth through staged size increases.
    """
    
    # Disable dynamic growth: single fixed stage -> no growth will occur
    GROWTH_STAGES = [(13, 13)]
    
    @staticmethod
    def _get_available_compute():
        """Calculate available compute resources for dynamic scaling"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            return (100 - cpu_percent) / 100, (100 - memory_percent) / 100
        except ImportError:
            return 0.5, 0.5  # Default to middle values if psutil not available
    
    def __init__(self,
                 map_size: tuple = (13, 13),
                 input_dim: int = 64,
                 learning_rate: float = 0.8,
                 sigma: float = 7.0,
                 plasticity_decay: float = 0.02,
                 plasticity_mod: float = 0.6,
                 fatigue_cost: float = 0.5,
                 fatigue_decay: float = 0.03,
                 activation_threshold: float = 5.0,
                 inhibition_radius_factor: float = 2.0,
                 growth_threshold: float = 0.85,  # High utilization triggers growth
                 growth_cooldown: int = 1000,    # Minimum cycles between growth
                 inhibition_strength: float = 0.1):
        """
        Initializes the CAS SOM with parameters for emergent complexity and failure learning.

        Args:
            map_size (tuple): (height, width) of the neuron grid.
            input_dim (int): Dimensionality of the input vectors.
            learning_rate (float): Initial global learning rate.
            sigma (float): Initial radius of the excitatory neighborhood.
            plasticity_decay (float): Rate at which plasticity decays towards baseline.
            plasticity_mod (float): Amount plasticity increases after activation.
            fatigue_cost (float): Energy lost by BMU per learning event.
            fatigue_decay (float): Rate of energy regeneration for neurons.
            activation_threshold (float): Max distance for a neuron to be a BMU.
            inhibition_radius_factor (float): Multiplier for inhibitory radius vs. excitatory sigma.
            inhibition_strength (float): Magnitude of lateral inhibition.
        """
        self.map_height = np.int32(map_size[0])
        self.map_width = np.int32(map_size[1])
        self.map_size = (self.map_height, self.map_width)
        self.input_dim = np.int32(input_dim)
        
        # --- Core Parameters ---
        self.initial_lr = np.float32(learning_rate)
        self.initial_sigma = np.float32(sigma)
        self.activation_threshold = np.float32(activation_threshold)
        self.inhibition_radius_factor = np.float32(inhibition_radius_factor) # Stored factor, not fixed radius
        self.inhibition_strength = np.float32(inhibition_strength)
        
        # --- Plasticity Configuration ---
        self.plasticity_baseline = np.float32(1.0)
        self.plasticity_decay = np.float32(plasticity_decay)
        self.plasticity_mod = np.float32(plasticity_mod)

        # --- Fatigue Configuration ---
        self.fatigue_baseline = np.float32(1.0)
        self.fatigue_cost = np.float32(fatigue_cost)
        self.fatigue_decay = np.float32(fatigue_decay)

        # --- Internal State Matrices ---
        # Initialize weights with small random values within [-1, 1]
        self.weights = np.random.uniform(-1, 1, (self.map_height, self.map_width, self.input_dim)).astype(np.float32)
        # Plasticity initialized to baseline
        self.plasticity = np.full(self.map_size, self.plasticity_baseline, dtype=np.float32)
        # Fatigue initialized to baseline (fully charged)
        self.fatigue = np.full(self.map_size, self.fatigue_baseline, dtype=np.float32)
        
        # --- Pre-computed Static Data ---
        # Neuron coordinates for distance calculations
        self.neuron_coords = np.indices(self.map_size).transpose(1, 2, 0).astype(np.float32)
        self.last_update_time = time.perf_counter() # For fatigue decay calculation

        # --- Growth Parameters ---
        self.growth_threshold = growth_threshold
        self.growth_cooldown = growth_cooldown
        self.current_growth_stage = 0  # Index into GROWTH_STAGES
        self.cycles_since_growth = 0
        self.utilization_history = []
        
        # --- Failure State Variables ---
        self.critical_threshold_exceeded = False # Flag for external systems
        self.failure_log = [] # Stores data upon critical state detection

        # Initialize logging data structures
        self.growth_log = []
        self.performance_metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'learning_times': [],
            'quantization_errors': [],
            'growth_events': [],
            'plasticity_stats': [],
            'fatigue_stats': [],
            'utilization_stats': []
        }
        
        # Log initialization
        init_log = {
            'timestamp': time.time(),
            'event': 'startup',
            'map_size': map_size,
            'input_dim': input_dim,
            'params': {
                'learning_rate': learning_rate,
                'sigma': sigma,
                'plasticity_decay': plasticity_decay,
                'fatigue_decay': fatigue_decay,
                'activation_threshold': activation_threshold
            }
        }
        self.growth_log.append(init_log)
        logger.info(f"CAS SOM started: Map={map_size}, InputDim={input_dim}")
    # Usage tracking structures will be initialized lazily on first learning call
    # (process_and_learn sets bmu_heatmap, neuron_age, last_activation as needed).

    def _log_performance_metrics(self):
        """Log current performance metrics"""
        cpu_avail, mem_avail = self._get_available_compute()
        current_metrics = {
            'timestamp': time.time(),
            'cpu_usage': 1.0 - cpu_avail,
            'memory_usage': 1.0 - mem_avail,
            'map_size': self.map_size,
            'quantization_error': np.mean(self.quantization_errors[-50:]) if hasattr(self, 'quantization_errors') and self.quantization_errors else None,
            'plasticity_mean': float(np.mean(self.plasticity)),
            'plasticity_std': float(np.std(self.plasticity)),
            'fatigue_mean': float(np.mean(self.fatigue)),
            'fatigue_std': float(np.std(self.fatigue)),
            'utilization': float(np.sum(self.bmu_heatmap > 0) / (self.map_height * self.map_width))
        }
        
        # Update metrics history
        for key, value in current_metrics.items():
            if key in self.performance_metrics:
                self.performance_metrics[key].append(value)
                
        # Keep only last 1000 entries for each metric
        for key in self.performance_metrics:
            if len(self.performance_metrics[key]) > 1000:
                self.performance_metrics[key] = self.performance_metrics[key][-1000:]
        
        return current_metrics

    def _check_growth_conditions(self) -> bool:
        """Check if the SOM should grow based on multiple factors and available resources"""
        if self.current_growth_stage >= len(self.GROWTH_STAGES) - 1:
            return False  # Already at maximum size
            
        if self.cycles_since_growth < self.growth_cooldown:
            return False  # Still in cooldown period
            
        # 1. Calculate recent utilization
        recent_utilization = np.mean(self.utilization_history[-100:]) if self.utilization_history else 0
        
        # 2. Check BMU distribution (clustering)
        bmu_distribution = self.bmu_heatmap / (np.sum(self.bmu_heatmap) + 1e-6)
        entropy = -np.sum(bmu_distribution * np.log2(bmu_distribution + 1e-6))
        max_entropy = np.log2(self.map_height * self.map_width)
        distribution_uniformity = entropy / max_entropy
        
        # 3. Check average quantization error trend
        recent_errors = self.quantization_errors[-50:] if hasattr(self, 'quantization_errors') else []
        error_trend = np.mean(np.diff(recent_errors)) if len(recent_errors) > 1 else 0
        
        # 4. Check neighborhood plasticity
        avg_plasticity = np.mean(self.plasticity)
        plasticity_variance = np.var(self.plasticity)
        
        # 5. Check available compute resources
        cpu_avail, mem_avail = self._get_available_compute()
        resource_score = min(cpu_avail, mem_avail)
        
        # Adaptive growth threshold based on resources
        adaptive_threshold = self.growth_threshold * (1.0 + (1.0 - resource_score))
        
        # Growth conditions with resource awareness:
        needs_growth = (
            resource_score > 0.3 and (  # Only grow if sufficient resources
                (recent_utilization > adaptive_threshold) or              # High utilization (adaptive)
                (distribution_uniformity < 0.7 and avg_plasticity < 0.3 and resource_score > 0.5) or  # Poor distribution
                (error_trend > 0 and recent_utilization > adaptive_threshold * 0.8) or  # Rising errors
                (plasticity_variance < 0.1 and recent_utilization > adaptive_threshold * 0.9)  # Uniform plasticity
            )
        )
        
        return needs_growth

    def _grow_map(self):
        """Grow the SOM to the next size stage while preserving learned patterns"""
        if not self._check_growth_conditions():
            return
            
        old_size = self.map_size
        new_size = self.GROWTH_STAGES[self.current_growth_stage + 1]
        
        # Store old weights and states
        old_weights = self.weights.copy()
        old_plasticity = self.plasticity.copy()
        old_fatigue = self.fatigue.copy()
        
        # Initialize new arrays
        self.map_height, self.map_width = new_size
        self.map_size = new_size
        self.weights = np.random.uniform(-1, 1, (self.map_height, self.map_width, self.input_dim)).astype(np.float32)
        self.plasticity = np.full(self.map_size, self.plasticity_baseline, dtype=np.float32)
        self.fatigue = np.full(self.map_size, self.fatigue_baseline, dtype=np.float32)
        
        # Interpolate old values into new arrays
        y_indices = np.linspace(0, old_size[0]-1, new_size[0])
        x_indices = np.linspace(0, old_size[1]-1, new_size[1])
        for i, y in enumerate(y_indices):
            for j, x in enumerate(x_indices):
                y1, y2 = int(np.floor(y)), int(np.ceil(y))
                x1, x2 = int(np.floor(x)), int(np.ceil(x))
                
                if y1 == y2: y2 = min(y1 + 1, old_size[0] - 1)
                if x1 == x2: x2 = min(x1 + 1, old_size[1] - 1)
                
                # Bilinear interpolation weights
                wy2 = y - y1
                wy1 = 1 - wy2
                wx2 = x - x1
                wx1 = 1 - wx2
                
                # Interpolate weights
                self.weights[i, j] = (wy1 * (wx1 * old_weights[y1, x1] + wx2 * old_weights[y1, x2]) +
                                    wy2 * (wx1 * old_weights[y2, x1] + wx2 * old_weights[y2, x2]))
                                    
                # Interpolate plasticity and fatigue
                self.plasticity[i, j] = (wy1 * (wx1 * old_plasticity[y1, x1] + wx2 * old_plasticity[y1, x2]) +
                                       wy2 * (wx1 * old_plasticity[y2, x1] + wx2 * old_plasticity[y2, x2]))
                self.fatigue[i, j] = (wy1 * (wx1 * old_fatigue[y1, x1] + wx2 * old_fatigue[y1, x2]) +
                                    wy2 * (wx1 * old_fatigue[y2, x1] + wx2 * old_fatigue[y2, x2]))
        
        # Update coordinates for distance calculations
        self.neuron_coords = np.indices(self.map_size).transpose(1, 2, 0).astype(np.float32)
        
        # Update growth state
        self.current_growth_stage += 1
        self.cycles_since_growth = 0
        
        # Initialize tracking for new size
        self.bmu_heatmap = np.zeros(self.map_size, dtype=np.float32)
        self.neuron_age = np.zeros(self.map_size, dtype=np.float32)
        self.last_activation = np.full(self.map_size, -np.inf, dtype=np.float32)
        
        # Log growth event with detailed metrics
        growth_event = {
            'timestamp': time.time(),
            'event': 'growth',
            'old_size': old_size,
            'new_size': new_size,
            'metrics_before': self._log_performance_metrics(),
            'growth_stage': self.current_growth_stage,
            'resource_state': {
                'cpu_available': self._get_available_compute()[0],
                'memory_available': self._get_available_compute()[1]
            }
        }
        self.growth_log.append(growth_event)
        self.performance_metrics['growth_events'].append(growth_event)
        
        logger.info(f"SOM grew from {old_size} to {new_size}")

    def _prune_unused_neurons(self, current_time: float):
        """Identify and reset unused or ineffective neurons"""
        # Calculate time since last activation for each neuron
        activation_age = current_time - self.last_activation
        
        # Identify neurons that haven't been active recently
        inactive_mask = activation_age > (self.growth_cooldown * 0.5)
        low_utility_mask = self.bmu_heatmap < np.mean(self.bmu_heatmap) * 0.1
        
        neurons_to_reset = inactive_mask & low_utility_mask
        
        if np.any(neurons_to_reset):
            # Reset these neurons with slight variations of neighboring active neurons
            active_indices = np.where(~neurons_to_reset)
            reset_indices = np.where(neurons_to_reset)
            
            for reset_y, reset_x in zip(*reset_indices):
                # Find nearest active neuron
                distances = np.sqrt(
                    (active_indices[0] - reset_y)**2 + 
                    (active_indices[1] - reset_x)**2
                )
                nearest_idx = np.argmin(distances)
                nearest_y, nearest_x = active_indices[0][nearest_idx], active_indices[1][nearest_idx]
                
                # Reset with variation
                self.weights[reset_y, reset_x] = (
                    self.weights[nearest_y, nearest_x] + 
                    np.random.normal(0, 0.1, self.input_dim)
                )
                self.plasticity[reset_y, reset_x] = self.plasticity_baseline
                self.fatigue[reset_y, reset_x] = self.fatigue_baseline
                self.neuron_age[reset_y, reset_x] = 0
                
            logger.info(f"Reset {np.sum(neurons_to_reset)} unused neurons")

    def _find_bmu(self, input_vector: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Finds the Best Matching Unit (BMU), considering neuronal fatigue. 
        Returns None if no neuron is a sufficiently good match (below threshold).
        """
        # Fatigue integration: fatigued neurons (low fatigue value) are less responsive.
        # We model this by increasing their effective distance, making them less likely BMUs.
        # Add a small epsilon to prevent division by zero if fatigue is exactly 0.
        fatigue_penalty = 1.0 / (self.fatigue + 1e-6) 
        
        # Calculate squared Euclidean distance between input and all weights
        distance_sq = np.sum(np.square(self.weights - input_vector.reshape((1, 1, self.input_dim))), axis=2)
        
        # Apply fatigue penalty to distances
        fatigue_adjusted_distance = distance_sq * fatigue_penalty

        # Find the index of the minimum adjusted distance
        min_dist_idx = np.argmin(fatigue_adjusted_distance)
        bmu_coords = np.unravel_index(min_dist_idx, self.map_size)
        # Convert to builtin ints for consistency
        bmu_coords = (int(bmu_coords[0]), int(bmu_coords[1]))

        # Activation Threshold: If the best match is still too far, no effective BMU.
        # Use the unadjusted distance for the threshold check.
        if np.sqrt(distance_sq[bmu_coords]) > self.activation_threshold:
            return None

        return bmu_coords

    def _update_internal_states(self, bmu_coords: tuple, excitatory_influence: np.ndarray):
        """
        Updates dynamic internal state matrices (fatigue, plasticity) and handles
        temporal decay for lingering SOM vectors. Called after a learning event.
        """
        time_delta = time.perf_counter() - self.last_update_time
        
        # --- Update Fatigue ---
        # All neurons regenerate energy over time
        self.fatigue += self.fatigue_decay * time_delta
        # The winning neuron (if any) expends energy
        if bmu_coords is not None:
            self.fatigue[bmu_coords] -= self.fatigue_cost
        # Clip fatigue to prevent negative values and excessive energy
        self.fatigue = np.clip(self.fatigue, 0.01, self.fatigue_baseline)

        # --- Update Plasticity ---
        # Plasticity decays towards baseline for all neurons
        self.plasticity *= (1.0 - self.plasticity_decay * time_delta) # Scale decay by time
        # Neurons influenced by learning become more plastic
        if bmu_coords is not None:
            plasticity_increase = excitatory_influence * self.plasticity_mod
            self.plasticity += plasticity_increase
        # Clip plasticity to prevent extreme values
        self.plasticity = np.clip(self.plasticity, 0.1, 5.0)

        # --- Temporal Decay on Lingering SOM Vectors ---
        # The decay of plasticity and increase in fatigue on less active neurons
        # serve as implicit temporal decay. No additional explicit decay of weights themselves
        # is added here, as it's typically handled by the learning rule and neighborhood function
        # making weights of inactive neurons less influenced over time.
        
        self.last_update_time = time.perf_counter()

    def _check_and_log_failure(self, input_vector: np.ndarray, bmu_coords: tuple):
        """Checks for failure conditions and logs data if a critical threshold is met."""
        
        # Condition 1: BMU fatigue is critically low
        if bmu_coords is not None and self.fatigue[bmu_coords] < 0.01:  # Lower threshold; avoid frequent triggers
            # Log the condition but do not set the global critical suppression flag.
            logger.warning(f"Low BMU fatigue detected for {bmu_coords}: {self.fatigue[bmu_coords]:.4f}")

        # Condition 2: Very poor match despite finding a BMU (input is very far from learned patterns)
        if bmu_coords is not None:
            bmu_distance = np.sqrt(np.sum(np.square(self.weights[bmu_coords] - input_vector)**2))
            if bmu_distance > self.activation_threshold * 1.5: # 1.5x the normal activation threshold
                # Log the condition for diagnostics but do not force a critical suppression.
                logger.warning(f"Poor BMU match at {bmu_coords} (dist={bmu_distance:.2f}); continuing operation.")
        
        # Condition 3: No BMU found at all (input is completely novel or unmappable)
        if bmu_coords is None:
            logger.warning("No Best Matching Unit found for input. Input may be novel or unmappable; continuing.")

    def visualize_growth(self) -> dict:
        """Generate visualization data for the SOM's state"""
        current_time = time.perf_counter()
        
        viz_data = {
            'map_size': self.map_size,
            'utilization': {
                'current': np.sum(self.bmu_heatmap > 0) / (self.map_height * self.map_width),
                'history': self.utilization_history[-100:] if self.utilization_history else []
            },
            'neuron_health': {
                'activity_heatmap': self.bmu_heatmap.copy(),
                'age_map': self.neuron_age.copy(),
                'plasticity_map': self.plasticity.copy(),
                'fatigue_map': self.fatigue.copy()
            },
            'growth_metrics': {
                'stage': self.current_growth_stage,
                'next_size': self.GROWTH_STAGES[min(self.current_growth_stage + 1, len(self.GROWTH_STAGES) - 1)],
                'cycles_since_growth': self.cycles_since_growth,
                'growth_readiness': self._calculate_growth_readiness()
            }
        }
        return viz_data
        
    def _calculate_growth_readiness(self) -> float:
        """Calculate a normalized score (0-1) indicating readiness for growth"""
        if self.current_growth_stage >= len(self.GROWTH_STAGES) - 1:
            return 0.0
            
        recent_utilization = np.mean(self.utilization_history[-100:]) if self.utilization_history else 0
        time_factor = min(self.cycles_since_growth / self.growth_cooldown, 1.0)
        plasticity_health = np.mean(self.plasticity) / self.plasticity_baseline
        
        readiness = (
            0.4 * recent_utilization +
            0.3 * time_factor +
            0.3 * plasticity_health
        )
        return float(readiness)

    def process_and_learn(self, data_batch: list, learning_bias: float = 1.0) -> np.ndarray:
        """
        Processes a batch of data, performs learning, and returns activation map.
        Handles 'productive failures' by logging state and flagging the critical condition.
        Learning for an input is skipped if a critical state is detected for that input.
        """
        if not data_batch:
            return np.zeros(self.map_size, dtype=np.float32)

        # Lazy-init usage tracking structures if missing (backward compatibility)
        if not hasattr(self, 'bmu_heatmap'):
            self.bmu_heatmap = np.zeros(self.map_size, dtype=np.float32)
        if not hasattr(self, 'neuron_age'):
            self.neuron_age = np.zeros(self.map_size, dtype=np.float32)
        if not hasattr(self, 'last_activation'):
            self.last_activation = np.full(self.map_size, -np.inf, dtype=np.float32)

        iterations = len(data_batch)
        activation_map = np.zeros(self.map_size, dtype=np.float32)
        current_time = time.perf_counter()
        
        # The critical_threshold_exceeded flag should be reset by an *external* recovery mechanism
        # after it processes the failure. It reflects the current internal state of SOM.

        for i, input_vector in enumerate(data_batch):
            input_vector = input_vector.astype(np.float32)
            
            # --- Find BMU and Check for Failure ---
            bmu_coords = self._find_bmu(input_vector)
            self._check_and_log_failure(input_vector, bmu_coords)

            # If a critical state is detected or no BMU found, skip learning but update time-based states
            if bmu_coords is None or self.critical_threshold_exceeded:
                self._update_internal_states(bmu_coords, np.zeros(self.map_size))
                continue

            # --- Valid BMU found, proceed with learning ---
            # Calculate learning parameters that decay over the batch
            sigma_decayed = self.initial_sigma * (1.0 - i / iterations)  # Linear decay
            
            # Calculate distances and influences
            distance_sq = np.sum(np.square(self.neuron_coords - np.array(bmu_coords)), axis=2)
            
            # Calculate complex neighborhood functions
            excitatory_influence = np.exp(-distance_sq / (2 * np.square(sigma_decayed)))
            inhibitory_sigma = sigma_decayed * self.inhibition_radius_factor
            inhibitory_influence = self.inhibition_strength * np.exp(-distance_sq / (2 * np.square(inhibitory_sigma)))
            
            # Compute plasticity-modulated learning influence
            plasticity_influence = self.plasticity * (excitatory_influence - inhibitory_influence)
            
            # Calculate adaptive learning rate based on multiple factors
            local_lr = self.initial_lr * learning_bias * (
                (1.0 - i/iterations) *  # Time-based decay
                np.clip(self.fatigue[bmu_coords], 0.1, 1.0) *  # Fatigue modulation
                (1.0 + self.plasticity[bmu_coords] * 0.2)  # Plasticity boost
            )
            
            # Update weights with complex learning rule
            weight_delta = (
                local_lr * 
                plasticity_influence[..., np.newaxis] * 
                (input_vector - self.weights)
            )
            self.weights += weight_delta.astype(np.float32)
            
            # Update internal states and record activation
            self._update_internal_states(bmu_coords, excitatory_influence)
            activation_map[bmu_coords] += 1

        # Update neuron usage statistics
        self.bmu_heatmap += activation_map
        active_mask = activation_map > 0
        self.last_activation[active_mask] = current_time
        self.neuron_age += (current_time - self.last_update_time)
        
        # Calculate utilization for this batch
        utilization = np.sum(activation_map > 0) / (self.map_height * self.map_width)
        self.utilization_history.append(utilization)
        if len(self.utilization_history) > 1000:  # Keep last 1000 batches
            self.utilization_history.pop(0)
            
        # Track quantization errors
        if not hasattr(self, 'quantization_errors'):
            self.quantization_errors = []
        if len(data_batch) > 0:
            errors = [np.min(np.linalg.norm(self.weights - x.reshape(1, 1, -1), axis=2))
                     for x in data_batch]
            self.quantization_errors.append(np.mean(errors))
            if len(self.quantization_errors) > 1000:
                self.quantization_errors.pop(0)
        
        # Prune unused neurons periodically
        if self.cycles_since_growth % 100 == 0:
            self._prune_unused_neurons(current_time)
            
        # Increment cycles and check for growth
        self.cycles_since_growth += 1
        if self._check_growth_conditions():
            self._grow_map()
            
        self.last_update_time = current_time
        return activation_map
            
        return activation_map
        
    def get_activation_map(self, data_batch: list) -> np.ndarray:
        """Calculates the activation pattern of the SOM for a given batch without learning."""
        activation_map = np.zeros(self.map_size, dtype=np.float32)
        if not data_batch: return activation_map
        
        # Reset the critical_threshold_exceeded flag for this inference pass
        self.critical_threshold_exceeded = False 

        for vector in data_batch:
            vector = vector.astype(np.float32)
            bmu_coords = self._find_bmu(vector)
            # For pure activation map generation, we don't log failures or check critical state
            # but we still need to update internal states (like fatigue for future BMU finding)
            self._update_internal_states(bmu_coords, np.zeros(self.map_size)) # No learning, so zero influence
            if bmu_coords is not None:
                activation_map[bmu_coords] += 1
        return activation_map
        
                # if 'weights' in state and np.array(state['weights']).shape == self.weights.shape:
                #     self.weights = np.array(state['weights'], dtype=np.float32)
                   
                   
            
    def get_plasticity_map(self) -> np.ndarray:
        """Returns the current plasticity values for every neuron."""
        return self.plasticity.copy()
        
    def get_fatigue_map(self) -> np.ndarray:
        """Returns the current fatigue (energy) level for every neuron."""
        return self.fatigue.copy()
        
    def get_map_entropy(self) -> float:
        """Calculates the entropy of the map's weights, a measure of representation diversity."""
        # Flatten weights and calculate histogram for distribution analysis
        flat_weights = self.weights.flatten()
        # Use density=True to get probabilities, then remove zeros
        hist, _ = np.histogram(flat_weights, bins=256, density=True)
        hist = hist[hist > 0] # Remove zero probabilities to avoid log(0)
        
        # Shannon entropy calculation: -sum(p * log2(p))
        entropy = -np.sum(hist * np.log2(hist))
        return entropy
        
    def get_bmu_map(self, data_batch: list) -> np.ndarray:
        """Generates a map indicating how many times each neuron was a BMU for the data batch."""
        bmu_map = np.zeros(self.map_size, dtype=np.int32)
        if not data_batch: return bmu_map
        
        for vector in data_batch:
            vector = vector.astype(np.float32)
            bmu_coords = self._find_bmu(vector)
            if bmu_coords is not None:
                bmu_map[bmu_coords] += 1
        return bmu_map

    def get_quantization_error(self, data_batch: list) -> float:
        """Calculates the average distance between input vectors and their BMU weights."""
        total_error, n_samples = 0.0, 0
        if not data_batch: return 0.0
        
        for vector in data_batch:
            vector = vector.astype(np.float32)
            bmu_coords = self._find_bmu(vector)
            if bmu_coords is not None:
                bmu_weight = self.weights[bmu_coords]
                # Calculate Euclidean distance
                error = np.linalg.norm(vector - bmu_weight)
                total_error += error
                n_samples += 1
        
        # Return average error, or 0 if no samples mapped
        return (total_error / n_samples) if n_samples > 0 else 0.0

    # ---------------- Training Status / Introspection ----------------
    def get_training_status(self) -> dict:
        """Heuristic assessment of whether the SOM has been meaningfully trained.

        A SOM is considered 'trained' (heuristically) if:
          - We have a BMU heatmap with sufficient total hits ( > 5 * number of neurons )
          - Utilization (fraction of neurons that have been BMU at least once) exceeds 20%
          - Recent quantization error metric exists (was computed during learning cycles)
        These thresholds are adjustable and meant only for runtime sanity checks / logging.
        """
        total_neurons = int(self.map_height * self.map_width)
        total_bmu_hits = float(np.sum(self.bmu_heatmap)) if hasattr(self, 'bmu_heatmap') else 0.0
        utilization = float(
            np.sum(self.bmu_heatmap > 0) / total_neurons
        ) if hasattr(self, 'bmu_heatmap') and total_neurons > 0 else 0.0
        recent_qe = float(np.mean(self.quantization_errors[-10:])) if hasattr(self, 'quantization_errors') and self.quantization_errors else None
        trained = (
            total_bmu_hits > (5 * total_neurons) and
            utilization > 0.20 and
            recent_qe is not None
        )
        return {
            'trained': trained,
            'total_bmu_hits': total_bmu_hits,
            'utilization_fraction': utilization,
            'recent_quant_error': recent_qe,
            'map_size': self.map_size,
            'input_dim': int(self.input_dim),
            'neuron_count': total_neurons
        }

    # --- Backward Compatibility Adapter ---
    def process_input(self, input_vector: np.ndarray) -> np.ndarray:
        """Legacy single-vector processing interface expected by main.py.

        Accepts a 1D feature vector, performs a learning step (single item batch)
        using the existing process_and_learn path to keep internal statistics
        consistent, then returns a flattened activation map (map_height*map_width,).

        If the input is invalid it returns a zero activation vector.
        """
        try:
            if input_vector is None:
                logger.error("SOM.process_input received None as input_vector. Returning zero activation vector.")
                return np.zeros(self.map_height * self.map_width, dtype=np.float32)
            # Ensure 1D float32
            input_vec = np.array(input_vector, dtype=np.float32).reshape(-1)
            # Basic dimension guard – if mismatch, attempt truncate/pad
            if input_vec.shape[0] != self.input_dim:
                logger.warning(f"SOM.process_input received input_vector with mismatched dimensions. Expected {self.input_dim}, got {input_vec.shape[0]}.")
                if input_vec.shape[0] > self.input_dim:
                    input_vec = input_vec[:self.input_dim]
                else:
                    input_vec = np.pad(input_vec, (0, self.input_dim - input_vec.shape[0]))
            activation_map = self.process_and_learn([input_vec])
            return activation_map.astype(np.float32).flatten()
        except Exception as e:
            logger.exception("Exception in SOM.process_input: {e}")
            return np.zeros(self.map_height * self.map_width, dtype=np.float32)






import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import time
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnergyState:
    cognitive_energy: float = 1.0  # Mental processing power
    physical_energy: float = 1.0   # Physical action capability
    recovery_rate: float = 1.0     # How fast energy replenishes
    stress_level: float = 0.0      # Affects energy consumption

@dataclass
class ResourceState:
    memory_allocation: float = 0.0  # Neural resource usage
    processing_load: float = 0.0    # Computational load
    attention_resources: float = 1.0 # Available attention
    learning_capacity: float = 1.0   # Current ability to learn

class Metabolism:
    """
    Manages energy distribution,this is a more advanced fatigue management than original.need 
     to add decresaseexhaustion or maturity scale
       resource allocation, and system maintenance.
    Simulates biological metabolism for cognitive and physical resources.
    """
    def __init__(self):
        # Core energy pools
        self.base_energy = 1.0
        self.max_energy = 1.0
        self.current_energy = EnergyState()
        
        # Resource tracking
        self.resources = ResourceState()
        
        # Activity costs (energy per second)
        self.energy_costs = {
            'thinking': 0.02,
            'learning': 0.05,
            'physical_action': 0.03,
            'emotional_processing': 0.02,
            'memory_access': 0.01,
            'dream_state': 0.01
        }
        
        # Performance curves (how performance scales with energy)
        self.performance_curves = {
            'cognitive': lambda e: 1.0 - 0.8 * np.exp(-2.0 * e),
            'physical': lambda e: 1.0 - 0.9 * np.exp(-2.5 * e),
            'emotional': lambda e: 1.0 - 0.7 * np.exp(-1.8 * e)
        }
        
        # System state
        self.last_update = time.time()
        self.total_uptime = 0.0
        self.maintenance_needed = False
        self.dream_cycles = 0
        
        # Adaptive parameters
        self.adaptation_rate = 0.1
        self.stress_tolerance = 0.7
        self.recovery_bonus = 1.0
        
        # Resource pools (for different types of processing)
        self.resource_pools = {
            'short_term_memory': 1.0,
            'working_memory': 1.0,
            'attention_focus': 1.0,
            'emotional_capacity': 1.0,
            'learning_buffer': 1.0,
            'prediction_capacity': 1.0,
        }
        
        # Track burned calories (as metaphor for processing cost)
        self.calories_burned = 0.0
        self.calorie_history = []

    def update(self, dt: float):
        """Update metabolic state"""
        self.total_uptime += dt
        
        # Natural energy recovery
        self._process_energy_recovery(dt)
        
        # Resource reallocation
        self._manage_resources(dt)
        
        # Check for maintenance needs
        self._check_maintenance_status()
        
        # Update adaptation parameters
        self._adapt_parameters(dt)
        
        # Track calorie burn
        self._update_calorie_tracking(dt)

    def _process_energy_recovery(self, dt: float):
        """Handle natural energy recovery and limits"""
        # Base recovery rate affected by stress
        effective_recovery = self.current_energy.recovery_rate * \
                           (1.0 - 0.5 * self.current_energy.stress_level)
        
        # Apply recovery
        self.current_energy.cognitive_energy = min(
            self.max_energy,
            self.current_energy.cognitive_energy + effective_recovery * dt * 0.1
        )
        self.current_energy.physical_energy = min(
            self.max_energy,
            self.current_energy.physical_energy + effective_recovery * dt * 0.08
        )

    def _manage_resources(self, dt: float):
        """Manage and reallocate resources based on needs"""
        # Decay resource usage
        self.resources.memory_allocation *= 0.95
        self.resources.processing_load *= 0.90
        
        # Recover attention resources
        self.resources.attention_resources = min(
            1.0,
            self.resources.attention_resources + dt * 0.1
        )
        
        # Update learning capacity based on energy and stress
        self.resources.learning_capacity = self.performance_curves['cognitive'](
            self.current_energy.cognitive_energy
        ) * (1.0 - 0.5 * self.current_energy.stress_level)

    def _check_maintenance_status(self):
        """Check if system needs maintenance (like sleep)"""
        # Trigger maintenance need if:
        # - Energy too low
        # - Too much uptime
        # - Resource depletion
        energy_low = min(self.current_energy.cognitive_energy,
                        self.current_energy.physical_energy) < 0.3
        long_uptime = self.total_uptime > 16 * 3600  # 16 hours
        resources_low = min(self.resource_pools.values()) < 0.3
        
        self.maintenance_needed = energy_low or long_uptime or resources_low

    def _adapt_parameters(self, dt: float):
        """Adapt metabolic parameters based on usage patterns"""
        # Increase stress tolerance with exposure
        if self.current_energy.stress_level > self.stress_tolerance:
            self.stress_tolerance = min(
                0.9,
                self.stress_tolerance + dt * self.adaptation_rate
            )
        
        # Adjust recovery bonus based on maintenance cycles
        if self.dream_cycles > 0:
            self.recovery_bonus = min(
                2.0,
                self.recovery_bonus + 0.1 * self.dream_cycles
            )

    def _update_calorie_tracking(self, dt: float):
        """Track calorie burn as metaphor for processing cost"""
        # Base metabolic rate
        base_burn = 0.1 * dt  # 0.1 calories per second base rate
        
        # Additional costs based on activity
        cognitive_cost = (1.0 - self.current_energy.cognitive_energy) * 0.2 * dt
        physical_cost = (1.0 - self.current_energy.physical_energy) * 0.3 * dt
        stress_cost = self.current_energy.stress_level * 0.1 * dt
        
        total_burn = base_burn + cognitive_cost + physical_cost + stress_cost
        self.calories_burned += total_burn
        
        # Keep history for analysis
        self.calorie_history.append((time.time(), total_burn))
        if len(self.calorie_history) > 1000:
            self.calorie_history.pop(0)

    def consume_energy(self, activity: str, intensity: float = 1.0) -> bool:
        """Attempt to consume energy for an activity"""
        if activity not in self.energy_costs:
            return False
            
        cost = self.energy_costs[activity] * intensity
        
        # Check if we have enough energy
        if activity in ['thinking', 'learning', 'memory_access']:
            if self.current_energy.cognitive_energy < cost:
                return False
            self.current_energy.cognitive_energy -= cost
        else:
            if self.current_energy.physical_energy < cost:
                return False
            self.current_energy.physical_energy -= cost
            
        return True

    def allocate_resources(self, resource_type: str, amount: float) -> bool:
        """Attempt to allocate resources of a specific type"""
        if resource_type not in self.resource_pools:
            return False
            
        if self.resource_pools[resource_type] < amount:
            return False
            
        self.resource_pools[resource_type] -= amount
        return True

    def dream_cycle_completed(self):
        """Called when a dream cycle completes"""
        self.dream_cycles += 1
        self.maintenance_needed = False
        self.total_uptime = 0.0  # Reset uptime
        
        # Boost recovery and clear stress
        self.current_energy.recovery_rate = self.recovery_bonus
        self.current_energy.stress_level *= 0.5
        
        # Replenish resource pools
        for pool in self.resource_pools:
            self.resource_pools[pool] = min(1.0, self.resource_pools[pool] + 0.3)

    def get_performance_multiplier(self, activity_type: str) -> float:
        """Get current performance level for an activity type"""
        if activity_type in self.performance_curves:
            energy = (self.current_energy.cognitive_energy 
                     if activity_type == 'cognitive'
                     else self.current_energy.physical_energy)
            return self.performance_curves[activity_type](energy)
        return 1.0

    def get_status_report(self) -> Dict[str, float]:
        """Get current metabolic status"""
        return {
            'cognitive_energy': self.current_energy.cognitive_energy,
            'physical_energy': self.current_energy.physical_energy,
            'stress_level': self.current_energy.stress_level,
            'recovery_rate': self.current_energy.recovery_rate,
            'maintenance_needed': float(self.maintenance_needed),
            'calories_burned': self.calories_burned,
            'average_burn_rate': np.mean([b for _, b in self.calorie_history[-100:]])
        }
# JustinJ_Optimizer.py
# Standalone copy of JustinJOptimizer with internal audio feedback loop scaffolding.
# Retained original version still inside nn.py; this file allows portability & future divergence.

import numpy as np
import logging
import time
from collections import deque
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

class JustinJOptimizer:
    """Agency-focused optimizer with integrated vocal/audio internal feedback loop.

    Core Additions vs baseline in nn.py:
    - Optional internal echo loop if real mic input missing (still prefers real mic)
    - Audio alignment metrics (correlation, spectral centroid alignment)
    - Latent intention reconstruction (predict intended control vector from audio echo)
    - Multi-timescale adaptive LR modulation
    - Gradient hygiene (clipping + nan/inf scrubbing)
    - Pattern memory with decay + replay sampling hooks
    """
    def __init__(self,
                 networks: List,
                 base_lr: float = 1e-4,
                 vocal_feedback_weight: float = 0.3,
                 agency_growth_rate: float = 0.01,
                 control_precision: float = 0.1,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 weight_decay: float = 0.01,
                 enable_internal_echo: bool = True,
                 spectral_weight: float = 0.2,
                 replay_capacity: int = 512,
                 fast_mode: bool = False):
                # Core setup (internal AdamW-like state; fully decoupled from external optimizer)
                self.networks = networks or []
                self.base_lr = float(base_lr)
                self.beta1 = float(beta1)
                self.beta2 = float(beta2)
                self.epsilon = float(epsilon)
                self.weight_decay = float(weight_decay)  # decoupled AdamW style
                self.vocal_feedback_weight = float(vocal_feedback_weight)
                self.agency_growth_rate = float(agency_growth_rate)
                self.control_precision = float(control_precision)
                self.enable_internal_echo = enable_internal_echo
                self.spectral_weight = float(spectral_weight)
                self.replay_capacity = replay_capacity
                self.fast_mode = bool(fast_mode)

                # Agency metrics
                self.voice_control_confidence = 0.0
                self.intention_alignment = 0.0
                self.feedback_recognition = 0.0
                self.spectral_alignment = 0.0

                # Adaptive learning buffers
                self.p = {}
                self.t = 0

                # Memories
                self.short_term_memory = []
                self.pattern_memory = []
                self.replay_buffer = []

                # Gradient safety
                self.max_grad_norm = 1.0
                self.clip_gradients = True

                # Cached last intended output for echo fallback
                self._last_intended_output = None
                self._last_audio_feedback = None

                # Extended adaptive scheduling
                self.warmup_steps = 500
                self.cooldown_patience = 800
                self.min_lr = self.base_lr * 0.1
                self.max_lr = self.base_lr * 5.0
                self.last_improvement_step = 0
                self.improvement_threshold = 0.002
                self.metric_history = deque(maxlen=400)

                # Temporal coherence tracking
                self._ema_vocal = None
                self.temporal_coherence = 0.0

                # Additional spectral stats
                self.spectral_flatness = 0.0
                self.spectral_bandwidth = 0.0

                # Prioritized replay
                self.priority_replay = []
                self.priority_alpha = 0.7
                self.priority_epsilon = 1e-4

                # Plateau detection buffers
                self._rolling_vc = deque(maxlen=100)
                self._rolling_alignment = deque(maxlen=100)

                # Gradient freeze map
                self.frozen_params = {}
                self.freeze_duration = 300
                self.freeze_threshold = 0.0005
                self.max_freeze_fraction = 0.15

                # Moment & variance state per parameter key
                self.m = {}
                self.v = {}
                self._bias_correction_cache = {}

                logger.info("JustinJOptimizer (standalone) initialized (enriched + integrated AdamW).")

    # --------------------------- Audio / Feedback Utilities ---------------------------
    def _normalize_audio(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        m = np.max(np.abs(x)) + 1e-8
        return x / m

    def _spectral_centroid(self, signal: np.ndarray, sr: int = 44100) -> float:
        signal = signal.astype(np.float32)
        if signal.size == 0:
            return 0.0
        # FFT
        spec = np.fft.rfft(signal)
        mag = np.abs(spec)
        freqs = np.fft.rfftfreq(signal.size, 1.0 / sr)
        denom = np.sum(mag) + 1e-8
        return float(np.sum(freqs * mag) / denom)

    def _spectral_flatness_bandwidth(self, signal: np.ndarray, sr: int = 44100) -> Tuple[float, float]:
        signal = signal.astype(np.float32)
        if signal.size == 0:
            return 0.0, 0.0
        spec = np.fft.rfft(signal)
        mag = np.abs(spec) + 1e-12
        geo_mean = np.exp(np.mean(np.log(mag)))
        arith_mean = np.mean(mag)
        flatness = float(geo_mean / (arith_mean + 1e-12))
        freqs = np.fft.rfftfreq(signal.size, 1.0 / sr)
        centroid = self._spectral_centroid(signal, sr)
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / (np.sum(mag) + 1e-12)))
        return flatness, bandwidth

    def _calculate_feedback_match(self, vocal_output: np.ndarray, audio_feedback: np.ndarray) -> float:
        v = self._normalize_audio(vocal_output).flatten()
        a = self._normalize_audio(audio_feedback).flatten()
        if v.size != a.size:
            n = min(v.size, a.size)
            v = v[:n]; a = a[:n]
        if v.size == 0:
            return 0.0
        corr = np.corrcoef(v, a)[0,1]
        if not np.isfinite(corr):
            corr = 0.0
        return max(0.0, float(corr))

    def _calculate_control_accuracy(self, intended: np.ndarray, actual: np.ndarray) -> float:
        if intended is None or actual is None or intended.size == 0 or actual.size == 0:
            return 0.0
        n = min(intended.size, actual.size)
        err = np.mean((intended[:n] - actual[:n])**2)
        return float(np.exp(-err / max(self.control_precision,1e-6)))

    def _spectral_alignment_metric(self, vocal_output: np.ndarray, audio_feedback: np.ndarray) -> float:
        c1 = self._spectral_centroid(vocal_output)
        c2 = self._spectral_centroid(audio_feedback)
        diff = abs(c1 - c2)
        return float(np.exp(-diff / 800.0))  # heuristic scale

    # --------------------------- Public Metric Update ---------------------------
    def update_metrics(self,
                       vocal_output: Optional[np.ndarray],
                       audio_feedback: Optional[np.ndarray],
                       intended_output: Optional[np.ndarray] = None,
                       reward_signal: Optional[float] = None,
                       sample_rate: int = 44100):
        # Fallback echo if no mic capture yet
        if (audio_feedback is None or audio_feedback.size == 0) and self.enable_internal_echo:
            audio_feedback = vocal_output.copy() if vocal_output is not None else self._last_audio_feedback
        if vocal_output is None and self._last_intended_output is not None:
            vocal_output = self._last_intended_output  # crude fallback

        if vocal_output is None or audio_feedback is None:
            return

        # Core correlations
        fb_match = self._calculate_feedback_match(vocal_output, audio_feedback)
        self.feedback_recognition = self.feedback_recognition * 0.95 + fb_match * 0.05

        if intended_output is not None:
            ctrl_acc = self._calculate_control_accuracy(intended_output, vocal_output)
            self.intention_alignment = self.intention_alignment * 0.9 + ctrl_acc * 0.1
        else:
            ctrl_acc = 0.0

        if not self.fast_mode:
            spec_align = self._spectral_alignment_metric(vocal_output, audio_feedback)
            self.spectral_alignment = self.spectral_alignment * 0.9 + spec_align * 0.1

            # Additional spectral stats
            flatness, bandwidth = self._spectral_flatness_bandwidth(vocal_output, sample_rate)
            self.spectral_flatness = 0.95 * self.spectral_flatness + 0.05 * flatness
            self.spectral_bandwidth = 0.95 * self.spectral_bandwidth + 0.05 * bandwidth
        else:
            spec_align = 0.0
            flatness = 0.0
            bandwidth = 0.0

        # Temporal coherence: similarity of current output to EMA
        if self._ema_vocal is None:
            self._ema_vocal = vocal_output.astype(np.float32)
        else:
            # update ema
            self._ema_vocal = 0.9 * self._ema_vocal + 0.1 * vocal_output.astype(np.float32)
            n = min(self._ema_vocal.size, vocal_output.size)
            if n > 0:
                num = np.dot(self._ema_vocal[:n], vocal_output[:n])
                den = (np.linalg.norm(self._ema_vocal[:n]) * np.linalg.norm(vocal_output[:n]) + 1e-8)
                coh = num / den
                if np.isfinite(coh):
                    self.temporal_coherence = 0.95 * self.temporal_coherence + 0.05 * coh

        self.voice_control_confidence = (
            0.5 * self.feedback_recognition + 0.3 * self.intention_alignment + 0.2 * self.spectral_alignment
        )

        context = {
            'time': time.time(),
            'vocal_output': vocal_output.copy(),
            'audio_feedback': audio_feedback.copy(),
            'intended_output': None if intended_output is None else intended_output.copy(),
            'reward': reward_signal,
            'fb_match': fb_match,
            'ctrl_acc': ctrl_acc,
            'spec_align': spec_align,
            'spec_flat': flatness,
            'spec_band': bandwidth,
            'temp_coh': self.temporal_coherence
        }
        self.short_term_memory.append(context)
        if len(self.short_term_memory) > 128:
            self.short_term_memory.pop(0)

        if reward_signal is not None and reward_signal > 0.7:
            self.pattern_memory.append(context)
            if len(self.pattern_memory) > 512:
                self.pattern_memory.pop(0)

        # Replay store (simple chronological)
        self.replay_buffer.append(context)
        if len(self.replay_buffer) > self.replay_capacity:
            self.replay_buffer.pop(0)

        # Priority replay (priority derived from combined metrics + reward)
        priority = (
            0.4 * fb_match + 0.3 * self.intention_alignment + 0.2 * spec_align + 0.1 * self.temporal_coherence
        )
        if reward_signal is not None:
            priority += 0.3 * reward_signal
        if not self.fast_mode:
            self.priority_replay.append((float(priority), context))
            if len(self.priority_replay) > self.replay_capacity:
                self.priority_replay.pop(0)

        # Track improvement history for schedule decisions
        self.metric_history.append(self.voice_control_confidence)
        self._rolling_vc.append(self.voice_control_confidence)
        self._rolling_alignment.append(self.intention_alignment)

        # Detect improvements
        if len(self.metric_history) > 5:
            recent = list(self.metric_history)[-5:]
            if (max(recent) - min(recent)) > self.improvement_threshold:
                self.last_improvement_step = self.t

        self._last_intended_output = intended_output.copy() if intended_output is not None else self._last_intended_output
        self._last_audio_feedback = audio_feedback.copy()

    # --------------------------- Optimization Step ---------------------------
    def step(self):
        """Perform one optimization step (integrated AdamW + adaptive agency scaling)."""
        self.t += 1

        # ----- Learning rate scheduling -----
        agency_factor = np.sqrt(self.voice_control_confidence + 0.1)
        schedule_lr = self.base_lr
        if self.t < self.warmup_steps:  # warmup
            warm_frac = self.t / max(1, self.warmup_steps)
            schedule_lr = self.base_lr * (0.1 + 0.9 * warm_frac)
        elif (self.t - self.last_improvement_step) > self.cooldown_patience:  # cooldown plateau
            schedule_lr = max(self.min_lr, self.base_lr * 0.5)
        elif self.voice_control_confidence > 0.8:  # expansion when strong
            schedule_lr = min(self.max_lr, self.base_lr * (1.0 + (self.voice_control_confidence - 0.8) * 2.0))

        effective_lr = schedule_lr * agency_factor

        # ----- Parameter loop -----
        freeze_keys_considered = 0
        for net in self.networks:
            for param_array, grad_name, layer_instance in net.get_trainable_params():
                grad_array = getattr(layer_instance, grad_name)
                if grad_array is None:
                    continue
                key = (id(net), id(layer_instance), grad_name)

                # Initialize per-param state lazily
                if key not in self.p:
                    self.p[key] = np.ones_like(param_array, dtype=np.float32)
                if key not in self.m:
                    self.m[key] = np.zeros_like(param_array, dtype=np.float32)
                if key not in self.v:
                    self.v[key] = np.zeros_like(param_array, dtype=np.float32)

                # Skip if currently frozen
                if key in self.frozen_params and self.t < self.frozen_params[key]:
                    continue

                # ----- Agency potential update -----
                if self.voice_control_confidence > self.p[key].mean():
                    self.p[key] *= (1.0 + self.agency_growth_rate)
                else:
                    self.p[key] *= (1.0 - 0.1 * self.agency_growth_rate)

                # Exploration vs consolidation scaling
                if len(self.pattern_memory) > 0:
                    scale = np.sqrt(self.p[key]) * 1.15
                else:
                    scale = np.sqrt(self.p[key]) * 0.85
                if self.temporal_coherence > 0.6:  # damp updates if already stable
                    scale *= 0.9

                grad_array = grad_array * scale
                grad_array = np.nan_to_num(grad_array, nan=0.0, posinf=0.0, neginf=0.0)

                # Lightweight inactivity freeze heuristic
                if freeze_keys_considered < 500:
                    mean_abs = float(np.mean(np.abs(grad_array)))
                    if mean_abs < self.freeze_threshold and len(self.frozen_params) < int(self.max_freeze_fraction * 1000):
                        self.frozen_params[key] = self.t + self.freeze_duration
                    freeze_keys_considered += 1

                # ----- AdamW moment updates -----
                m = self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad_array
                v = self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad_array * grad_array)
                # Bias corrections
                bc1 = 1 - self.beta1 ** self.t
                bc2 = 1 - self.beta2 ** self.t
                m_hat = m / (bc1 + 1e-12)
                v_hat = v / (bc2 + 1e-12)
                update = m_hat / (np.sqrt(v_hat) + self.epsilon)

                # Decoupled weight decay
                if self.weight_decay > 0.0:
                    param_array -= effective_lr * self.weight_decay * param_array

                # Apply update
                param_array -= effective_lr * update

        if self.clip_gradients:
            self._clip_gradients()

        if self.t % 200 == 0:
            logger.info(
                f"JustinJOpt t={self.t} lr_eff={effective_lr:.2e} VC={self.voice_control_confidence:.3f} "
                f"FB={self.feedback_recognition:.3f} IA={self.intention_alignment:.3f} SPEC={self.spectral_alignment:.3f} "
                f"Flat={self.spectral_flatness:.3f} Band={self.spectral_bandwidth:.1f} Coh={self.temporal_coherence:.3f} Replay={len(self.replay_buffer)}")

        # store last lr for diagnostics
        self._last_effective_lr = effective_lr

    # --------------------------- Prioritized Replay Utilities ---------------------------
    def _sample_priority_indices(self, batch_size: int) -> List[int]:
        if not self.priority_replay:
            return []
        priorities = np.array([p for p, _ in self.priority_replay], dtype=np.float32)
        probs = (priorities + self.priority_epsilon) ** self.priority_alpha
        probs /= probs.sum()
        count = min(batch_size, len(self.priority_replay))
        return list(np.random.choice(len(self.priority_replay), size=count, replace=False, p=probs))

    def optimize_with_replay(self, batch_size: int = 16):
        """Lightweight auxiliary adaptation using prioritized replay contexts.
        This does NOT backprop (no computational graph) but modulates agency potentials
        using stored high-priority alignment examples so future gradient scaling reflects them.
        """
        if self.fast_mode:
            return
        idxs = self._sample_priority_indices(batch_size)
        if not idxs:
            return
        avg_fb = 0.0; avg_ctrl = 0.0; avg_spec = 0.0
        for i in idxs:
            priority, ctx = self.priority_replay[i]
            avg_fb += ctx['fb_match']
            avg_ctrl += ctx['ctrl_acc']
            avg_spec += ctx['spec_align']
        n = len(idxs)
        avg_fb /= n; avg_ctrl /= n; avg_spec /= n
        composite = 0.5 * avg_fb + 0.3 * avg_ctrl + 0.2 * avg_spec
        # Adjust global potentials subtly
        adjust = (composite - 0.5) * 0.01  # small influence
        for k in list(self.p.keys())[:200]:  # cap cost
            self.p[k] *= (1.0 + adjust)

    # --------------------------- Diagnostics ---------------------------
    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            **self.get_agency_metrics(),
            'spectral_flatness': float(self.spectral_flatness),
            'spectral_bandwidth': float(self.spectral_bandwidth),
            'temporal_coherence': float(self.temporal_coherence),
            'fast_mode': self.fast_mode,
            'last_effective_lr': getattr(self, '_last_effective_lr', None),
            'lr_base': float(self.base_lr),
            'frozen_params': len(self.frozen_params),
            'warmup_remaining': max(0, self.warmup_steps - self.t),
            'since_improvement': self.t - self.last_improvement_step
        }

    # --------------------------- Gradient Clipping ---------------------------
    def _clip_gradients(self):
        total_norm = 0.0
        for net in self.networks:
            for _, grad_name, layer in net.get_trainable_params():
                g = getattr(layer, grad_name)
                if g is not None:
                    total_norm += float(np.linalg.norm(g)**2)
        total_norm = np.sqrt(total_norm)
        if total_norm > self.max_grad_norm:
            coef = self.max_grad_norm / (total_norm + 1e-6)
            for net in self.networks:
                for _, grad_name, layer in net.get_trainable_params():
                    g = getattr(layer, grad_name)
                    if g is not None:
                        setattr(layer, grad_name, g * coef)

    # --------------------------- Replay Sampling Hook ---------------------------
    def sample_replay(self, batch_size: int = 8) -> List[Dict[str, Any]]:
        if not self.replay_buffer:
            return []
        idxs = np.random.choice(len(self.replay_buffer), size=min(batch_size, len(self.replay_buffer)), replace=False)
        return [self.replay_buffer[i] for i in idxs]

    def get_agency_metrics(self) -> Dict[str, float]:
        return {
            'voice_control_confidence': float(self.voice_control_confidence),
            'intention_alignment': float(self.intention_alignment),
            'feedback_recognition': float(self.feedback_recognition),
            'spectral_alignment': float(self.spectral_alignment),
            'pattern_memory_size': len(self.pattern_memory),
            'replay_size': len(self.replay_buffer),
            'temporal_coherence': float(self.temporal_coherence)
        }

# EOF

# Utilities for dynamic/growable neural components (pure NumPy)
import numpy as np
import math
import logging
logger = logging.getLogger(__name__)

class GrowableLinear:
    """A growable linear layer (input_dim x output_dim) supporting runtime expansion.
    We keep weights as (in_dim, out_dim) like nn.Linear logic implemented in nn.py but without
    fatigue/plasticity. Used as an adapter when we must splice new capacity into existing models.
    """
    def __init__(self, input_dim:int, output_dim:int, activation=None, dtype=np.float32):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dtype = dtype
        limit = math.sqrt(6.0/(input_dim+output_dim))
        self.weights = np.random.uniform(-limit, limit, (input_dim, output_dim)).astype(dtype)
        self.biases = np.zeros((1, output_dim), dtype=dtype)
        self.activation = activation  # optional callable

    def forward(self, x:np.ndarray)->np.ndarray:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"GrowableLinear forward dim mismatch {x.shape[-1]} vs {self.input_dim}")
        y = x @ self.weights + self.biases
        if self.activation:
            y = self.activation(y)
        return y

    def expand_input(self, new_input_dim:int):
        if new_input_dim <= self.input_dim:
            return False
        add = new_input_dim - self.input_dim
        limit = math.sqrt(6.0/(new_input_dim + self.output_dim))
        extra = np.random.uniform(-limit, limit, (add, self.output_dim)).astype(self.dtype)
        self.weights = np.concatenate([self.weights, extra], axis=0)
        self.input_dim = new_input_dim
        return True

    def expand_output(self, new_output_dim:int):
        if new_output_dim <= self.output_dim:
            return False
        add = new_output_dim - self.output_dim
        limit = math.sqrt(6.0/(self.input_dim + new_output_dim))
        extra_w = np.random.uniform(-limit, limit, (self.input_dim, add)).astype(self.dtype)
        self.weights = np.concatenate([self.weights, extra_w], axis=1)
        extra_b = np.zeros((1, add), dtype=self.dtype)
        self.biases = np.concatenate([self.biases, extra_b], axis=1)
        self.output_dim = new_output_dim
        return True

class DimensionRegistry:
    """Central lightweight registry so modules can query and publish evolving dimensions.
    Not thread-safe; orchestrator mutates before/after cycle boundaries.
    """
    #64dim through 8192dim and <34000 dim emotional_processing, +,- <68100 dim sfe/cafve dual convolute SOM_BMU_COORD_DIM
    def __init__(self):
        self._dims = {}
    def publish(self, name:str, value:int):
        self._dims[name] = int(value)
    def get(self, name:str, default=None):
        return self._dims.get(name, default)
    def snapshot(self):
        return dict(self._dims)

# Utility functions for dimension handling int2 through float64 dynamic quantizations

def pad_or_truncate(vec: np.ndarray, target: int) -> np.ndarray:
    """Return a 1D float64 vector of length target by zero-padding or truncating.
    Accepts None -> zeros.
    """
    if vec is None:
        return np.zeros(target, dtype=np.float64)
    v = np.array(vec, dtype=np.float64).reshape(-1)
    n = v.shape[0]
    if n == target:
        return v
    if n > target:
        return v[:target]
    out = np.zeros(target, dtype=np.float64)
    out[:n] = v
    return out

def ensure_2d(row_vec: np.ndarray) -> np.ndarray:
    v = np.array(row_vec, dtype=np.float64)
    if v.ndim == 1:
        return v.reshape(1, -1)
    return v

def param_count_linear(in_dim: int, out_dim: int, bias: bool = True) -> int:
    return in_dim * out_dim + (out_dim if bias else 0)

def human(n: int) -> str:
    for unit in ['', 'K', 'M', 'B']:
        if n < 1000:
            return f"{n}{unit}"
        n //= 1000
    return f"{n}T"

# Global singleton (optional)
GLOBAL_DIM_REGISTRY = DimensionRegistry()



"""
organicnn.py

Single-file consolidation of all biological addon systems so user can relocate
and trim original scattered modules manually later. Originals are left intact.

Included subsystems (mirrors existing separate files):
  - IonChannel / DendriticBranch / Neuron (from bioneural.py)
  - BioSequential + BioDense + BioActivation  (from organic_nn.py)
  - EnergyState / ResourceState / Metabolism (from metabolism.py)
  - NeurotransmitterSystem (from neurotransmitter.py)
  - EndocrineSystem (from endocrine.py)
  - Reward (from reward.py)
  - PhysicalSensation (from sensation.py)
  - LayeredFatigue (from fatigue.py)
  - Unified BioSystem orchestrator (new) replacing bionet.py controller logic

No deletions performed. This file stands alone (no underscores in name) as the
monolithic biological substrate.
"""

from __future__ import annotations
import numpy as np
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IonChannel / DendriticBranch / Neuron  (from bioneural.py)
# ---------------------------------------------------------------------------
@dataclass
class IonChannel:
    channel_type: str
    conductance: float
    voltage_threshold: float
    activation_speed: float
    inactivation_rate: float
    current_state: float = 0.0
    recovery_time: float = 0.0

class DendriticBranch:
    def __init__(self, num_segments: int = 10):
        self.num_segments = num_segments
        self.voltage = np.zeros(num_segments, dtype=np.float32)
        self.calcium = np.zeros(num_segments, dtype=np.float32)
        self.synapses: List[List[Any]] = [[] for _ in range(num_segments)]
        self.spike_thresholds = np.full(num_segments, 0.7, dtype=np.float32)
        self.hotspots = np.random.choice([True, False], num_segments, p=[0.3, 0.7])

class Neuron:
    def __init__(self, num_dendrites: int = 5):
        self.resting_potential = -70.0
        self.threshold = -55.0
        self.reset_potential = -80.0
        self.membrane_potential = self.resting_potential
        self.ion_channels: Dict[str, IonChannel] = {
            'na_fast': IonChannel('sodium', 1.0, -55.0, 0.1, 0.5),
            'k_slow': IonChannel('potassium', 0.8, -65.0, 0.05, 0.1),
            'ca_persistent': IonChannel('calcium', 0.3, -50.0, 0.02, 0.05)
        }
        self.dendrites = [DendriticBranch() for _ in range(num_dendrites)]
        self.calcium_concentration = 0.0
        self.last_spike_time = 0.0
        self.refractory_period = 0.002

    def _update_ion_channels(self, dt: float):
        for ch in self.ion_channels.values():
            if self.membrane_potential >= ch.voltage_threshold:
                ch.current_state = min(1.0, ch.current_state + ch.activation_speed * dt)
            else:
                ch.current_state = max(0.0, ch.current_state - ch.inactivation_rate * dt)
            if ch.current_state < 0.1:
                ch.recovery_time = max(0.0, ch.recovery_time - dt)

    def _process_dendrites(self, dt: float) -> float:
        total = 0.0
        for d in self.dendrites:
            for i in range(d.num_segments):
                syn_input = sum(s.weight * s.activation for s in d.synapses[i])
                d.voltage[i] += syn_input
                if d.hotspots[i] and d.voltage[i] > d.spike_thresholds[i]:
                    d.voltage[i] = 20.0
                    d.calcium[i] += 0.5
                d.voltage[i] *= 0.9
                total += d.voltage[i]
        return total

    def _update_membrane(self, dend_current: float, dt: float):
        na = self.ion_channels['na_fast'].current_state * self.ion_channels['na_fast'].conductance * (50.0 - self.membrane_potential)
        k = self.ion_channels['k_slow'].current_state * self.ion_channels['k_slow'].conductance * (-100.0 - self.membrane_potential)
        ca = self.ion_channels['ca_persistent'].current_state * self.ion_channels['ca_persistent'].conductance * (100.0 - self.membrane_potential)
        leak = 0.1 * (self.resting_potential - self.membrane_potential)
        self.membrane_potential += dt * (na + k + ca + leak + dend_current)

    def _maybe_spike(self, now: float):
        if now - self.last_spike_time < self.refractory_period:
            return
        if self.membrane_potential >= self.threshold:
            self.membrane_potential = self.reset_potential
            self.last_spike_time = now
            self.calcium_concentration += 0.5
            self.ion_channels['na_fast'].current_state = 0.0
            self.ion_channels['na_fast'].recovery_time = self.refractory_period

    def _update_calcium(self, dt: float):
        self.calcium_concentration *= (1.0 - 0.1 * dt)
        for d in self.dendrites:
            d.calcium *= (1.0 - 0.1 * dt)

    def add_synaptic_input(self, dendrite_idx: int, segment_idx: int, weight: float, activation: float):
        if 0 <= dendrite_idx < len(self.dendrites):
            d = self.dendrites[dendrite_idx]
            if 0 <= segment_idx < d.num_segments:
                d.synapses[segment_idx].append(type('Synapse', (), {'weight': weight, 'activation': activation}))

    def update(self, dt: float, current_time: float):
        self._update_ion_channels(dt)
        dend = self._process_dendrites(dt)
        self._update_membrane(dend, dt)
        self._maybe_spike(current_time)
        self._update_calcium(dt)

    def get_state(self) -> Dict[str, Any]:
        return {
            'membrane': self.membrane_potential,
            'calcium': self.calcium_concentration,
            'channels': {n: c.current_state for n, c in self.ion_channels.items()}
        }

# ---------------------------------------------------------------------------
# BioDense / BioActivation / BioSequential / BioAdamW  (from organic_nn.py)
# ---------------------------------------------------------------------------
class BioLayer:
    def forward(self, x: np.ndarray) -> np.ndarray: raise NotImplementedError
    def backward(self, g: np.ndarray) -> np.ndarray: raise NotImplementedError
    def get_trainable_params(self): return []

class BioDense(BioLayer):
    def __init__(self, input_size: int, output_size: int):
        self.neurons = [Neuron(num_dendrites=5) for _ in range(output_size)]
        self.weights = (np.random.randn(input_size, output_size).astype(np.float32) * np.sqrt(2./input_size))
        self.biases = np.zeros((1, output_size), dtype=np.float32)
        self.plasticity_strength = 0.01
        self.plasticity_decay = 0.999
        self.synaptic_scaling = np.ones_like(self.weights)
        self.adaptation = np.zeros(output_size, dtype=np.float32)
        self.adaptation_rate = 0.1
        self.adaptation_recovery = 0.05
        self.input: Optional[np.ndarray] = None
        self.output: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x.astype(np.float32)
        bsz = x.shape[0]
        out = np.zeros((bsz, len(self.neurons)), dtype=np.float32)
        for b in range(bsz):
            sample = self.input[b]
            for idx, neuron in enumerate(self.neurons):
                splits = np.array_split(sample, len(neuron.dendrites))
                for d_idx, (dendrite, d_input) in enumerate(zip(neuron.dendrites, splits)):
                    for i, val in enumerate(d_input):
                        w = self.weights[i + d_idx * len(d_input), idx]
                        neuron.add_synaptic_input(d_idx, i % dendrite.num_segments, w * self.synaptic_scaling[i + d_idx * len(d_input), idx], val)
                neuron.update(0.001, time.time())
                out[b, idx] = neuron.membrane_potential * (1.0 - self.adaptation[idx])
                if abs(out[b, idx]) > 0.1:
                    self.adaptation[idx] += self.adaptation_rate
                self.adaptation[idx] *= (1.0 - self.adaptation_recovery)
        self.output = out
        return out

    def backward(self, g: np.ndarray) -> np.ndarray:
        if self.input is None or self.output is None:
            return g
        adapt_fac = 1.0 - self.adaptation
        grad_scaled = g * adapt_fac
        w_grad = np.zeros_like(self.weights)
        for b in range(self.input.shape[0]):
            w_grad += np.outer(self.input[b], grad_scaled[b])
            pre_post = np.outer(self.input[b], self.output[b])
            w_grad += self.plasticity_strength * pre_post * self.synaptic_scaling
            self.synaptic_scaling *= self.plasticity_decay
            self.synaptic_scaling += (1 - self.plasticity_decay) * (pre_post > 0)
        w_grad /= self.input.shape[0]
        self.weights_gradient = w_grad
        self.biases_gradient = np.mean(grad_scaled, axis=0, keepdims=True)
        return np.dot(grad_scaled, self.weights.T)

    def get_trainable_params(self):
        return [(self.weights, 'weights_gradient', self), (self.biases, 'biases_gradient', self)]

class BioActivation(BioLayer):
    def __init__(self):
        self.threshold = -55.0
        self.refractory = 0.002
        self.last_spike = np.zeros(1)
        self.neurotx = np.ones(1)
        self.depletion = 0.2
        self.recover = 0.1
        self.input: Optional[np.ndarray] = None
        self.output: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        now = time.time()
        if self.last_spike.shape != x.shape:
            self.last_spike = np.zeros_like(x)
            self.neurotx = np.ones_like(x)
        t_since = now - self.last_spike
        spike = (x > self.threshold) & (t_since > self.refractory)
        self.last_spike[spike] = now
        out = np.zeros_like(x)
        out[spike] = 1.0
        out *= self.neurotx
        self.neurotx[spike] *= (1.0 - self.depletion)
        self.neurotx += self.recover * (1.0 - self.neurotx)
        self.output = out
        return out

    def backward(self, g: np.ndarray) -> np.ndarray:
        if self.input is None: return g
        deriv = (self.input > self.threshold).astype(np.float32)
        return g * deriv * self.neurotx

class BioSequential:
    def __init__(self, *layers: BioLayer):
        self.layers = layers
    def forward(self, x: np.ndarray) -> np.ndarray:
        for l in self.layers: x = l.forward(x)
        return x
    def backward(self, g: np.ndarray) -> np.ndarray:
        for l in reversed(self.layers): g = l.backward(g)
        return g
    def get_trainable_params(self):
        params = []
        for l in self.layers:
            if hasattr(l, 'get_trainable_params'):
                params.extend(l.get_trainable_params())
        return params

def bio_mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_pred_scaled = (y_pred + 70) / 115
    return np.mean((y_pred_scaled - y_true)**2).astype(np.float32)

def bio_mse_loss_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_pred_scaled = (y_pred + 70) / 115
    return 2 * (y_pred_scaled - y_true) / (y_true.size * 115)

"""
class BioAdamW:
    def __init__(self, nets: List[BioSequential], lr: float = 1e-3, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01):
        self.lr = np.float32(lr); self.b1=np.float32(beta1); self.b2=np.float32(beta2); self.eps=np.float32(eps); self.wd=np.float32(wd)
        self.nets = nets; self.m={}; self.v={}; self.t=0
        self.max_w=5.0; self.min_w=-5.0; self.home_target=0.1; self.home_rate=0.001
    def step(self):
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.b2**self.t)/(1 - self.b1**self.t)
        for net in self.nets:
            for p, grad_name, layer in net.get_trainable_params():
                grad = getattr(layer, grad_name, None)
                if grad is None: continue
                key = (id(net), id(layer), grad_name)
                if key not in self.m:
                    self.m[key]=np.zeros_like(p); self.v[key]=np.zeros_like(p)
                self.m[key]=self.b1*self.m[key]+(1-self.b1)*grad
                self.v[key]=self.b2*self.v[key]+(1-self.b2)*(grad**2)
                upd = lr_t * self.m[key]/(np.sqrt(self.v[key])+self.eps)
                p -= (upd + self.lr*self.wd*p)
                if grad_name=='weights_gradient':
                    np.clip(p, self.min_w, self.max_w, out=p)
                    if isinstance(layer, BioDense):
                        act = np.mean(np.abs(layer.output)>0.1)
                        p *= (1.0 + self.home_rate*(self.home_target-act))
                        """

# ---------------------------------------------------------------------------
# Metabolism (from metabolism.py)
# ---------------------------------------------------------------------------
@dataclass
class EnergyState:
    cognitive_energy: float = 1.0
    physical_energy: float = 1.0
    recovery_rate: float = 1.0
    stress_level: float = 0.0

@dataclass
class ResourceState:
    memory_allocation: float = 0.0
    processing_load: float = 0.0
    attention_resources: float = 1.0
    learning_capacity: float = 1.0

class Metabolism:
    def __init__(self):
        self.base_energy=1.0; self.max_energy=1.0
        self.current_energy=EnergyState()
        self.resources=ResourceState()
        self.energy_costs={'thinking':0.02,'learning':0.05,'physical_action':0.03,'emotional_processing':0.02,'memory_access':0.01,'dream_state':0.01}
        self.performance_curves={'cognitive':lambda e:1.0-0.8*np.exp(-2.0*e),'physical':lambda e:1.0-0.9*np.exp(-2.5*e),'emotional':lambda e:1.0-0.7*np.exp(-1.8*e)}
        self.last_update=time.time(); self.total_uptime=0.0; self.maintenance_needed=False; self.dream_cycles=0
        self.adaptation_rate=0.1; self.stress_tolerance=0.7; self.recovery_bonus=1.0
        self.resource_pools={'short_term_memory':1.0,'working_memory':1.0,'attention_focus':1.0,'emotional_capacity':1.0,'learning_buffer':1.0}
        self.calories_burned=0.0; self.calorie_history=[]
    def update(self, dt: float):
        self.total_uptime+=dt; self._process_energy_recovery(dt); self._manage_resources(dt); self._check_maintenance_status(); self._adapt_parameters(dt); self._update_calorie_tracking(dt)
    def _process_energy_recovery(self, dt: float):
        eff = self.current_energy.recovery_rate*(1.0-0.5*self.current_energy.stress_level)
        self.current_energy.cognitive_energy=min(self.max_energy,self.current_energy.cognitive_energy+eff*dt*0.1)
        self.current_energy.physical_energy=min(self.max_energy,self.current_energy.physical_energy+eff*dt*0.08)
    def _manage_resources(self, dt: float):
        self.resources.memory_allocation*=0.95; self.resources.processing_load*=0.90
        self.resources.attention_resources=min(1.0,self.resources.attention_resources+dt*0.1)
        self.resources.learning_capacity=self.performance_curves['cognitive'](self.current_energy.cognitive_energy)*(1.0-0.5*self.current_energy.stress_level)
    def _check_maintenance_status(self):
        energy_low=min(self.current_energy.cognitive_energy,self.current_energy.physical_energy)<0.3
        long_uptime=self.total_uptime>16*3600
        resources_low=min(self.resource_pools.values())<0.3
        self.maintenance_needed=energy_low or long_uptime or resources_low
    def _adapt_parameters(self, dt: float):
        if self.current_energy.stress_level>self.stress_tolerance:
            self.stress_tolerance=min(0.9,self.stress_tolerance+dt*self.adaptation_rate)
        if self.dream_cycles>0:
            self.recovery_bonus=min(2.0,self.recovery_bonus+0.1*self.dream_cycles)
    def _update_calorie_tracking(self, dt: float):
        base=0.1*dt; cog=(1.0-self.current_energy.cognitive_energy)*0.2*dt; phys=(1.0-self.current_energy.physical_energy)*0.3*dt; stress=self.current_energy.stress_level*0.1*dt
        burn=base+cog+phys+stress; self.calories_burned+=burn; self.calorie_history.append((time.time(),burn));
        if len(self.calorie_history)>1000: self.calorie_history.pop(0)
    def consume_energy(self, activity: str, intensity: float=1.0)->bool:
        if activity not in self.energy_costs: return False
        cost=self.energy_costs[activity]*intensity
        pool='cognitive_energy' if activity in ['thinking','learning','memory_access'] else 'physical_energy'
        if getattr(self.current_energy,pool)<cost: return False
        setattr(self.current_energy,pool,getattr(self.current_energy,pool)-cost); return True
    def dream_cycle_completed(self):
        self.dream_cycles+=1; self.maintenance_needed=False; self.total_uptime=0.0; self.current_energy.recovery_rate=self.recovery_bonus; self.current_energy.stress_level*=0.5
        for k in self.resource_pools: self.resource_pools[k]=min(1.0,self.resource_pools[k]+0.3)
    def get_status_report(self)->Dict[str,float]:
        return {'cognitive_energy':self.current_energy.cognitive_energy,'physical_energy':self.current_energy.physical_energy,'stress_level':self.current_energy.stress_level,'recovery_rate':self.current_energy.recovery_rate,'maintenance_needed':float(self.maintenance_needed),'calories_burned':self.calories_burned,'average_burn_rate':np.mean([b for _,b in self.calorie_history[-100:]]) if self.calorie_history else 0.0}

# ---------------------------------------------------------------------------
# Neurotransmitter System (from neurotransmitter.py) condensed
# ---------------------------------------------------------------------------
@dataclass
class Neurotransmitter:
    name: str; baseline: float=0.5; current_level: float=0.5; decay_rate: float=0.1; synthesis_rate: float=0.1; reuptake_rate: float=0.2; depletion_threshold: float=0.2; max_level: float=1.0

class NeurotransmitterSystem:
    def __init__(self):
        self.transmitters={
            'glutamate':Neurotransmitter('glutamate',baseline=0.4,decay_rate=0.15,synthesis_rate=0.2),
            'gaba':Neurotransmitter('gaba',baseline=0.4,decay_rate=0.12,synthesis_rate=0.15),
            'dopamine':Neurotransmitter('dopamine',baseline=0.3,decay_rate=0.1,synthesis_rate=0.08),
            'serotonin':Neurotransmitter('serotonin',baseline=0.5,decay_rate=0.05,synthesis_rate=0.06),
            'norepinephrine':Neurotransmitter('norepinephrine',baseline=0.3,decay_rate=0.12,synthesis_rate=0.1),
            'acetylcholine':Neurotransmitter('acetylcholine',baseline=0.4,decay_rate=0.08,synthesis_rate=0.1),
        }
        self.receptor_sensitivity={n:1.0 for n in self.transmitters}
        self.precursor_levels={n:1.0 for n in self.transmitters}
        self.vesicle_pools={n:0.8 for n in self.transmitters}
        self.reuptake_efficiency={n:1.0 for n in self.transmitters}
        self.recent_activity={n:[] for n in self.transmitters}; self.max_history=1000
        names=list(self.transmitters.keys()); self.interaction_matrix=np.zeros((len(names),len(names)),dtype=np.float32)
        interactions={('dopamine','serotonin'):-0.2,('serotonin','dopamine'):-0.1,('glutamate','gaba'):0.3,('gaba','glutamate'):-0.4,('norepinephrine','gaba'):-0.2,('acetylcholine','glutamate'):0.2,('dopamine','acetylcholine'):0.2,('serotonin','norepinephrine'):-0.1}
        for (a,b),val in interactions.items():
            i=names.index(a); j=names.index(b); self.interaction_matrix[i,j]=val
    def update(self, dt: float, neural_activity: Optional[Dict[str,float]]=None):
        if neural_activity is None: neural_activity={n:0.0 for n in self.transmitters}
        for name,t in self.transmitters.items():
            act=neural_activity.get(name,0.0); release=act*self.vesicle_pools[name]*t.current_level
            self.vesicle_pools[name]=max(0.0,self.vesicle_pools[name]-release+t.synthesis_rate*dt)
            reuptake=(t.current_level-t.baseline)*t.reuptake_rate*self.reuptake_efficiency[name]*dt
            decay=t.current_level*t.decay_rate*dt
            synthesis=t.synthesis_rate*self.precursor_levels[name]*dt
            t.current_level=np.clip(t.current_level+synthesis-decay-reuptake,0.0,t.max_level)
            self.recent_activity[name].append(act)
            if len(self.recent_activity[name])>self.max_history: self.recent_activity[name].pop(0)
            avg=np.mean(self.recent_activity[name][-100:]); target=0.3; sens_change=(target-avg)*0.01*dt
            self.receptor_sensitivity[name]=np.clip(self.receptor_sensitivity[name]+sens_change,0.5,2.0)
        self._process_interactions(dt); self._update_precursors(dt)
    def _process_interactions(self, dt: float):
        names=list(self.transmitters.keys()); levels=np.array([self.transmitters[n].current_level for n in names])
        effects=self.interaction_matrix@levels
        for i,n in enumerate(names):
            self.transmitters[n].current_level=np.clip(self.transmitters[n].current_level+effects[i]*dt,0.0,self.transmitters[n].max_level)
    def _update_precursors(self, dt: float):
        for n in self.transmitters:
            regen=(1.0-self.precursor_levels[n])*0.1*dt; cons=self.transmitters[n].synthesis_rate*dt
            self.precursor_levels[n]=np.clip(self.precursor_levels[n]+regen-cons,0.1,1.0)
    def get_effective_levels(self)->Dict[str,float]:
        return {n:self.transmitters[n].current_level*self.receptor_sensitivity[n] for n in self.transmitters}

# ---------------------------------------------------------------------------
# Endocrine System condensed
# ---------------------------------------------------------------------------
@dataclass
class HormoneProfile:
    dopamine: float=0.0; serotonin: float=0.5; oxytocin: float=0.0; cortisol: float=0.2; adrenaline: float=0.0; melatonin: float=0.0; endorphins: float=0.0; vasopressin: float=0.0

class EndocrineSystem:
    def __init__(self, base_dimension: int = 1024):
        self.hormone_states={h:np.zeros(128,dtype=np.float32) for h in ['dopamine','serotonin','oxytocin','cortisol','adrenaline','melatonin','endorphins','vasopressin']}
        self.hormone_states['serotonin'].fill(0.5)
        self.internal_clock=0.0; self.homeostatic_targets=HormoneProfile(dopamine=0.3,serotonin=0.6,oxytocin=0.2,cortisol=0.2,adrenaline=0.1,melatonin=0.1,endorphins=0.1,vasopressin=0.2)
        self.decay_rates={'dopamine':0.1,'serotonin':0.05,'oxytocin':0.15,'cortisol':0.08,'adrenaline':0.2,'melatonin':0.1,'endorphins':0.12,'vasopressin':0.07}
        self.interaction_matrix=self._init_matrix()
    def _init_matrix(self):
        hs=list(self.hormone_states.keys()); m=np.zeros((len(hs),len(hs)),dtype=np.float32)
        inter={('dopamine','serotonin'):0.2,('cortisol','serotonin'):-0.3,('adrenaline','cortisol'):0.4,('melatonin','cortisol'):-0.2,('oxytocin','cortisol'):-0.3,('endorphins','cortisol'):-0.2,('adrenaline','dopamine'):0.2,('oxytocin','vasopressin'):0.3}
        for (a,b),val in inter.items(): i=hs.index(a); j=hs.index(b); m[i,j]=val
        return m
    def update(self, dt: float):
        self.internal_clock=(self.internal_clock+dt/3600)%24; self._circadian(); self._interact(dt); self._homeostasis(dt)
    def _circadian(self):
        h=self.internal_clock; night=np.sin(((h-20)%24)*np.pi/12); self.hormone_states['melatonin']+=max(0,night)*0.1
        morning=np.sin(((h-8)%24)*np.pi/12); self.hormone_states['cortisol']+=max(0,morning)*0.1
        day=np.sin(((h-12)%24)*np.pi/12); self.hormone_states['serotonin']+=max(0,day)*0.05
    def _interact(self, dt: float):
        hs=list(self.hormone_states.keys()); levels=np.array([np.mean(self.hormone_states[h]) for h in hs]); effects=self.interaction_matrix@levels
        for i,h in enumerate(hs): self.hormone_states[h]+=effects[i]*dt; self.hormone_states[h]=np.clip(self.hormone_states[h],0,1)
    def _homeostasis(self, dt: float):
        for h,target in vars(self.homeostatic_targets).items():
            cur=self.hormone_states[h]; decay=self.decay_rates[h]*dt; delta=(target-np.mean(cur))*decay; self.hormone_states[h]+=delta*np.random.normal(1,0.1,cur.shape); self.hormone_states[h]=np.clip(self.hormone_states[h],0,1)
    def trigger_response(self, event: str, intensity: float):
        responses={'pleasure':{'dopamine':0.5,'serotonin':0.2,'endorphins':0.3},'stress':{'cortisol':0.6,'adrenaline':0.4,'serotonin':-0.2},'social':{'oxytocin':0.4,'vasopressin':0.3,'dopamine':0.2},'pain':{'endorphins':0.5,'cortisol':0.3,'adrenaline':0.4},'fear':{'adrenaline':0.7,'cortisol':0.5,'oxytocin':-0.2},'achievement':{'dopamine':0.6,'serotonin':0.3,'endorphins':0.2}}
        if event in responses:
            for h,chg in responses[event].items():
                self.hormone_states[h]+=chg*intensity*np.random.normal(1,0.1,128); self.hormone_states[h]=np.clip(self.hormone_states[h],0,1)
    def get_hormone_profile(self)->HormoneProfile:
        return HormoneProfile(**{h:float(np.mean(self.hormone_states[h])) for h in self.hormone_states})
    def get_hormone_vector(self)->np.ndarray:
        return np.concatenate(list(self.hormone_states.values())).astype(np.float32)

# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------
class Reward:
    def __init__(self, emotion_dim: int = 1024):
        self.pleasure_threshold=0.9; self.pain_threshold=0.3; self.reward_vector=np.zeros(emotion_dim,dtype=np.float32)
        self.pleasure_decay_rate=0.1; self.pain_decay_rate=0.05; self.current_intensity=0.0
    def compute_reward(self, goals_sat: float, health_score: float, predict_error: float)->np.ndarray:
        self.current_intensity*= (1.0 - self.pleasure_decay_rate)
        if goals_sat>self.pleasure_threshold and predict_error<0.1:
            self.current_intensity=1.0
        elif health_score<self.pain_threshold or predict_error>0.5:
            self.current_intensity=-1.0
        self.reward_vector.fill(self.current_intensity); return self.reward_vector
    def get_current_intensity(self)->float: return self.current_intensity

# ---------------------------------------------------------------------------
# Physical Sensation
# ---------------------------------------------------------------------------
class PhysicalSensation:
    def __init__(self, sensation_dim: int = 1024):
        self.temperature=np.zeros(64,dtype=np.float32); self.pressure=np.zeros(128,dtype=np.float32); self.internal=np.zeros(256,dtype=np.float32); self.proprioception=np.zeros(576,dtype=np.float32)
        self.pain_threshold=0.7; self.discomfort_threshold=0.5; self.pleasure_threshold=0.8
        self.energy_level=1.0; self.comfort_level=1.0; self.last_movement_time=0.0
        self.optimal_temperature=0.5; self.optimal_energy=0.8
    def get_unified_sensation(self)->np.ndarray:
        return np.concatenate([self.temperature,self.pressure,self.internal,self.proprioception]).astype(np.float32)

# ---------------------------------------------------------------------------
# Layered Fatigue
# ---------------------------------------------------------------------------
class LayeredFatigue:
    def __init__(self, shape: Tuple[int,int], syn_recover: float=0.03, syn_cost: float=0.5, meta_decay: float=0.002, meta_rec_awake: float=0.0004, meta_rec_dream: float=0.01, floor: float=0.05):
        self.syn=np.full(shape,1.0,dtype=np.float32); self.meta=np.full(shape,1.0,dtype=np.float32)
        self.syn_recover=np.float32(syn_recover); self.syn_cost=np.float32(syn_cost); self.meta_decay=np.float32(meta_decay); self.meta_rec_awake=np.float32(meta_rec_awake); self.meta_rec_dream=np.float32(meta_rec_dream); self.floor=np.float32(floor)
    def apply_activation(self, coord: Optional[Tuple[int,int]], neighborhood_mean: float):
        if coord is None: return
        self.syn[coord]-=self.syn_cost; self.meta[coord]-=self.meta_decay*(0.5+neighborhood_mean)
    def recover(self, dt: float, dream: bool):
        self.syn+=self.syn_recover*dt; self.meta+= (self.meta_rec_dream if dream else self.meta_rec_awake)*dt
        self.syn=np.clip(self.syn,0.01,1.0); self.meta=np.clip(self.meta,self.floor,1.0)
    def combined(self)->np.ndarray: return 0.5*self.syn+0.5*self.meta
    def stats(self): c=self.combined(); return {'syn_mean':float(self.syn.mean()),'meta_mean':float(self.meta.mean()),'combined_mean':float(c.mean())}

# ---------------------------------------------------------------------------
# Unified BioSystem Orchestrator
# ---------------------------------------------------------------------------
class BioSystem:
    def __init__(self, emotion_dim: int=1024, som_shape: Optional[Tuple[int,int]]=None, enable_fatigue: bool=True):
        self.metabolism=Metabolism(); self.neuro=NeurotransmitterSystem(); self.hormones=EndocrineSystem(emotion_dim); self.reward=Reward(emotion_dim); self.sense=PhysicalSensation()
        self.fatigue=LayeredFatigue(som_shape) if (enable_fatigue and som_shape is not None) else None
        self.last_reward=0.0; self.energy_gate=1.0; self.learning_drive=1.0; self.explore=0.0; self.stability=1.0; self.mood=0.0; self.arousal=0.0
    def update(self, dt: float, goals: Optional[np.ndarray]=None, pred_vec: Optional[np.ndarray]=None, novelty: float=0.0, health: float=1.0, dream: bool=False, cog: Optional[np.ndarray]=None):
        goals_mean=float(np.mean(goals)) if goals is not None and goals.size>0 else 0.0
        pred_mag=float(np.linalg.norm(pred_vec)) if pred_vec is not None else 0.0
        pred_norm= pred_mag/(np.sqrt(pred_vec.size)+1e-6) if pred_vec is not None else 0.0
        rvec=self.reward.compute_reward(goals_sat=goals_mean, health_score=health, predict_error=pred_norm); self.last_reward=self.reward.get_current_intensity()
        if self.last_reward>0.8: self.hormones.trigger_response('pleasure', self.last_reward)
        elif self.last_reward<-0.5: self.hormones.trigger_response('pain', abs(self.last_reward))
        elif pred_norm>0.6: self.hormones.trigger_response('stress', pred_norm)
        elif novelty>0.7: self.hormones.trigger_response('achievement', novelty)
        self.hormones.update(dt)
        profile=self.hormones.get_hormone_profile()
        neural_activity={'dopamine':max(0.0,self.last_reward)+novelty*0.3+profile.dopamine,'serotonin':profile.serotonin*(1.0-pred_norm*0.3),'glutamate':0.3+pred_norm*0.4,'gaba':0.3+(1.0-pred_norm)*0.2,'norepinephrine':novelty*0.5+pred_norm*0.4,'acetylcholine':goals_mean*0.4+(1.0-pred_norm)*0.2}
        self.neuro.update(dt, neural_activity)
        self.metabolism.current_energy.stress_level=np.clip(pred_norm*0.6+(1.0-goals_mean)*0.4,0.0,1.0)
        if dream:
            self.metabolism.current_energy.recovery_rate=1.5; self.metabolism.consume_energy('dream_state', intensity=dt)
        else:
            think=0.5 if cog is None else float(np.clip(np.mean(np.abs(cog))*2.0,0.1,1.0)); self.metabolism.consume_energy('thinking', intensity=think*dt)
        self.metabolism.update(dt)
        if self.fatigue is not None:
            if cog is not None: self.fatigue.apply_activation((0,0), float(np.mean(np.abs(cog))))
            self.fatigue.recover(dt, dream)
            comb=float(self.fatigue.combined().mean())
        else:
            comb=1.0
        energy_frac=self.metabolism.current_energy.cognitive_energy; self.energy_gate=float(np.clip(energy_frac*comb,0.0,1.0))
        unmet=1.0-goals_mean; self.learning_drive=float(np.clip(0.5+0.7*unmet+0.5*pred_norm,0.2,2.0))
        self.explore=float(np.clip(novelty*0.3+unmet*0.4,0.0,1.5))
        self.stability=float(np.clip(1.0-self.metabolism.current_energy.stress_level*0.7,0.1,1.0))
        self.mood=float(np.clip(self.last_reward*0.5+profile.serotonin*0.5,-1.0,1.0))
        eff=self.neuro.get_effective_levels(); self.arousal=float(np.clip(profile.adrenaline*0.6+eff['norepinephrine']*0.4,0.0,1.5))
        self._cached_vectors={'reward':rvec,'hormones':self.hormones.get_hormone_vector(),'sensation':self.sense.get_unified_sensation(),'neuro_effective':np.array(list(eff.values()),dtype=np.float32)}
    def modulators(self)->Dict[str,float]:
        return {'energy_gate':self.energy_gate,'learning_drive':self.learning_drive,'explore':self.explore,'stability':self.stability,'mood':self.mood,'arousal':self.arousal,'reward_intensity':self.last_reward}
    def feature_block(self)->np.ndarray:
        parts=[]; cv=getattr(self,'_cached_vectors',{})
        for key in ['reward','hormones','neuro_effective']:
            if key in cv: parts.append(cv[key])
        if 'sensation' in cv:
            sens=cv['sensation']; parts.append(np.array([float(np.mean(sens)),float(np.std(sens)),float(np.max(sens))],dtype=np.float32))
        parts.append(np.array(list(self.modulators().values()),dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)





import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

# Integrated version: original detailed neuron model PLUS BioSequential wrapper
# (migrated from former root-level bioneural.py to keep biological code isolated
#  from main pathway as requested). Root bioneural.py removed after merge.
try:
    from nn import JustinJOptimizer  # Optional optimizer integration
except Exception:  # pragma: no cover - optional dependency
    JustinJOptimizer = None  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IonChannel:
    """
    Models an ion channel in a neuron's membrane.
    Ion channels are like tiny gates that control the flow of ions (charged particles)
    in and out of neurons. They're crucial for generating electrical signals.
    """
    # Channel types: sodium (Na+), potassium (K+), calcium (Ca2+), chloride (Cl-)
    channel_type: str
    conductance: float  # How easily ions flow through (0-1)
    voltage_threshold: float  # Voltage at which channel activates
    activation_speed: float  # How quickly channel responds
    inactivation_rate: float  # How quickly channel closes
    current_state: float = 0.0  # Open (1) or Closed (0)
    recovery_time: float = 0.0  # Time until channel can open again

class DendriticBranch:
    """
    Models a dendrite branch - the tree-like structures that receive inputs in neurons.
    Dendrites don't just collect signals, they perform complex computations by:
    1. Combining inputs in nonlinear ways
    2. Having local "hotspots" of activity
    3. Implementing timing-dependent integration
    """
    def __init__(self, num_segments: int = 10):
        self.num_segments = num_segments
        
        # Membrane voltage along the branch
        self.voltage = np.zeros(num_segments, dtype=np.float32)
        
        # Calcium concentration (for local computation)
        self.calcium = np.zeros(num_segments, dtype=np.float32)
        
        # Synaptic inputs along branch
        self.synapses = [[] for _ in range(num_segments)]
        
        # Local spike thresholds
        self.spike_thresholds = np.full(num_segments, 0.7, dtype=np.float32)
        
        # Active hotspots (segments that can generate local spikes)
        self.hotspots = np.random.choice([True, False], num_segments, p=[0.3, 0.7])

class Neuron:
    """
    A biologically-detailed neuron model with:
    1. Ion channels that control electrical signaling
    2. Dendritic branches that process inputs
    3. Local and global computation
    4. Synaptic plasticity
    """
    def __init__(self, num_dendrites: int = 5):
        # Membrane properties
        self.resting_potential = -70.0  # mV
        self.threshold = -55.0          # mV
        self.reset_potential = -80.0    # mV
        self.membrane_potential = self.resting_potential
        
        # Ion Channels (different types with different properties)
        self.ion_channels = {
            'na_fast': IonChannel(  # Fast sodium channels (spike generation)
                channel_type='sodium',
                conductance=1.0,
                voltage_threshold=-55.0,
                activation_speed=0.1,
                inactivation_rate=0.5
            ),
            'k_slow': IonChannel(  # Slow potassium channels (repolarization)
                channel_type='potassium',
                conductance=0.8,
                voltage_threshold=-65.0,
                activation_speed=0.05,
                inactivation_rate=0.1
            ),
            'ca_persistent': IonChannel(  # Calcium channels (synaptic plasticity)
                channel_type='calcium',
                conductance=0.3,
                voltage_threshold=-50.0,
                activation_speed=0.02,
                inactivation_rate=0.05
            )
        }
        
        # Dendritic tree (multiple branches)
        self.dendrites = [DendriticBranch() for _ in range(num_dendrites)]
        
        # Synaptic integration properties
        self.synaptic_inputs = []
        self.calcium_concentration = 0.0
        self.last_spike_time = 0.0
        self.refractory_period = 0.002  # 2ms
        
    def update(self, dt: float, current_time: float):
        """Update neuron state for one timestep"""
        # 1. Update ion channels
        self._update_ion_channels(dt)
        
        # 2. Process dendritic computation
        dendritic_current = self._process_dendrites(dt)
        
        # 3. Update membrane potential
        self._update_membrane_potential(dendritic_current, dt)
        
        # 4. Check for spike generation
        if self._check_spike_generation(current_time):
            self._generate_spike()
            
        # 5. Update calcium dynamics
        self._update_calcium(dt)

    def _update_ion_channels(self, dt: float):
        """Update all ion channel states"""
        for channel in self.ion_channels.values():
            # Check if voltage threshold is crossed
            if self.membrane_potential >= channel.voltage_threshold:
                # Activate channel based on its activation speed
                channel.current_state = min(1.0,
                    channel.current_state + channel.activation_speed * dt)
            else:
                # Inactivate channel
                channel.current_state = max(0.0,
                    channel.current_state - channel.inactivation_rate * dt)
            
            # Update recovery time
            if channel.current_state < 0.1:
                channel.recovery_time = max(0.0, channel.recovery_time - dt)

    def _process_dendrites(self, dt: float) -> float:
        """Process dendritic computation and return total dendritic current"""
        total_current = 0.0
        
        for dendrite in self.dendrites:
            # Propagate voltage along dendrite
            for i in range(dendrite.num_segments):
                # Local synaptic integration
                synaptic_input = sum(synapse.weight * synapse.activation 
                                   for synapse in dendrite.synapses[i])
                
                # Update segment voltage
                dendrite.voltage[i] += synaptic_input
                
                # Check for local spike generation in hotspots
                if dendrite.hotspots[i] and dendrite.voltage[i] > dendrite.spike_thresholds[i]:
                    # Generate local dendritic spike
                    dendrite.voltage[i] = 20.0  # Local spike
                    dendrite.calcium[i] += 0.5  # Calcium influx
                
                # Voltage decay
                dendrite.voltage[i] *= 0.9
                
                # Add to total current
                total_current += dendrite.voltage[i]
        
        return total_current

    def _update_membrane_potential(self, dendritic_current: float, dt: float):
        """Update membrane potential based on ion channels and dendritic input"""
        # Ion channel currents
        na_current = (self.ion_channels['na_fast'].current_state * 
                     self.ion_channels['na_fast'].conductance * 
                     (50.0 - self.membrane_potential))  # Sodium reversal potential
        
        k_current = (self.ion_channels['k_slow'].current_state * 
                    self.ion_channels['k_slow'].conductance * 
                    (-100.0 - self.membrane_potential))  # Potassium reversal potential
        
        ca_current = (self.ion_channels['ca_persistent'].current_state * 
                     self.ion_channels['ca_persistent'].conductance * 
                     (100.0 - self.membrane_potential))  # Calcium reversal potential
        
        # Leak current (passive return to rest)
        leak_current = 0.1 * (self.resting_potential - self.membrane_potential)
        
        # Update membrane potential
        self.membrane_potential += dt * (
            na_current + k_current + ca_current + leak_current + dendritic_current
        )

    def _check_spike_generation(self, current_time: float) -> bool:
        """Check if conditions are right for spike generation"""
        # Must be past refractory period
        if current_time - self.last_spike_time < self.refractory_period:
            return False
            
        # Check threshold crossing
        return self.membrane_potential >= self.threshold

    def _generate_spike(self):
        """Generate an action potential"""
        # Reset membrane potential
        self.membrane_potential = self.reset_potential
        
        # Record spike time
        self.last_spike_time = time.time()
        
        # Trigger calcium influx
        self.calcium_concentration += 0.5
        
        # Inactivate sodium channels (refractory period)
        self.ion_channels['na_fast'].current_state = 0.0
        self.ion_channels['na_fast'].recovery_time = self.refractory_period

    def _update_calcium(self, dt: float):
        """Update calcium concentration"""
        # Natural decay of calcium
        self.calcium_concentration *= (1.0 - 0.1 * dt)
        
        # Update calcium in dendrites
        for dendrite in self.dendrites:
            dendrite.calcium *= (1.0 - 0.1 * dt)

    def add_synaptic_input(self, dendrite_idx: int, segment_idx: int, 
                          weight: float, activation: float):
        """Add a synaptic input to a specific dendritic location"""
        if 0 <= dendrite_idx < len(self.dendrites):
            if 0 <= segment_idx < self.dendrites[dendrite_idx].num_segments:
                self.dendrites[dendrite_idx].synapses[segment_idx].append(
                    type('Synapse', (), {'weight': weight, 'activation': activation})
                )

    def get_state(self) -> Dict:
        """Get current state of the neuron"""
        return {
            'membrane_potential': self.membrane_potential,
            'calcium_concentration': self.calcium_concentration,
            'ion_channels': {name: channel.current_state 
                           for name, channel in self.ion_channels.items()},
            'dendrites': [{
                'voltage': dendrite.voltage.copy(),
                'calcium': dendrite.calcium.copy(),
                'num_synapses': [len(s) for s in dendrite.synapses]
            } for dendrite in self.dendrites]
        }


class BioSequential:
    """Lightweight sequential container with optional agency optimizer (merged)."""
    def __init__(self, *layers: Any):
        self.layers: List[Any] = list(layers)
        self.agency_optimizer: Optional[Any] = None

    def set_agency_optimizer(self, optimizer: Any):
        self.agency_optimizer = optimizer

    def forward(self, x):
        for layer in self.layers:
            if hasattr(layer, 'forward'):
                x = layer.forward(x)
            else:
                x = layer(x)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                grad_output = layer.backward(grad_output)
        return grad_output

    def update_agency_metrics(self, vocal_output, audio_feedback, intended_output=None, reward=None):
        if self.agency_optimizer and JustinJOptimizer is not None:
            try:
                self.agency_optimizer.update_metrics(vocal_output, audio_feedback, intended_output, reward)
            except Exception as e:  # pragma: no cover
                logger.warning(f"Agency metrics update failed: {e}")

    def agency_step(self):
        if self.agency_optimizer and JustinJOptimizer is not None:
            try:
                self.agency_optimizer.step()
            except Exception as e:  # pragma: no cover
                logger.warning(f"Agency step failed: {e}")

    def get_trainable_params(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'get_trainable_params'):
                params.extend(layer.get_trainable_params())
        return params




@dataclass
class HormoneProfile:
    dopamine: float = 0.0    # Reward/pleasure
    serotonin: float = 0.5   # Mood/wellbeing
    oxytocin: float = 0.0    # Bonding/trust
    cortisol: float = 0.2    # Stress/alertness
    adrenaline: float = 0.0  # Excitement/fear
    melatonin: float = 0.0   # Sleep/wake
    endorphins: float = 0.0  # Pain relief/euphoria
    vasopressin: float = 0.0 # Social bonding/memory

class EndocrineSystem:
    """
    complex hormone system that influences emotional, physical,
    and cognitive states through chemical messengers.  oh insulin now blurs the vision
    """
    def __init__(self, base_dimension: int = 1024):
        self.base_dimension = base_dimension
        
        # Hormone state vectors (each hormone has multiple subtypes/effects)
        self.hormone_states = {
            'dopamine': np.zeros(128, dtype=np.float32),
            'serotonin': np.full(128, 0.5, dtype=np.float32),
            'oxytocin': np.zeros(128, dtype=np.float32),
            'cortisol': np.zeros(128, dtype=np.float32),
            'adrenaline': np.zeros(128, dtype=np.float32),
            'melatonin': np.zeros(128, dtype=np.float32),
            'endorphins': np.zeros(128, dtype=np.float32),
            'vasopressin': np.zeros(128, dtype=np.float32)
        }
        
        # Hormone interaction matrix (how hormones affect each other)
        self.interaction_matrix = self._initialize_interaction_matrix()
        
        # Circadian rhythm tracking
        self.internal_clock = 0.0  # 0-24 hour cycle
        self.last_update = time.time()
        
        # Homeostatic targets (ideal hormone levels)
        self.homeostatic_targets = HormoneProfile(
            dopamine=0.3,
            serotonin=0.6,
            oxytocin=0.2,
            cortisol=0.2,
            adrenaline=0.1,
            melatonin=0.1,
            endorphins=0.1,
            vasopressin=0.2
        )
        
        # Recovery and decay rates
        self.decay_rates = {
            'dopamine': 0.1,
            'serotonin': 0.05,
            'oxytocin': 0.15,
            'cortisol': 0.08,
            'adrenaline': 0.2,
            'melatonin': 0.1,
            'endorphins': 0.12,
            'vasopressin': 0.07
        }

    def _initialize_interaction_matrix(self) -> np.ndarray:
        """Create matrix defining how hormones influence each other"""
        hormones = list(self.hormone_states.keys())
        n_hormones = len(hormones)
        matrix = np.zeros((n_hormones, n_hormones), dtype=np.float32)
        
        # Define hormone interactions (based on biological systems)
        interactions = {
            ('dopamine', 'serotonin'): 0.2,    # Dopamine boost affects serotonin
            ('cortisol', 'serotonin'): -0.3,   # Stress reduces serotonin
            ('adrenaline', 'cortisol'): 0.4,   # Adrenaline increases cortisol
            ('melatonin', 'cortisol'): -0.2,   # Sleep hormone reduces stress
            ('oxytocin', 'cortisol'): -0.3,    # Social bonding reduces stress
            ('endorphins', 'cortisol'): -0.2,  # Natural painkillers reduce stress
            ('adrenaline', 'dopamine'): 0.2,   # Excitement can be pleasurable
            ('oxytocin', 'vasopressin'): 0.3,  # Social hormones work together
        }
        
        for (h1, h2), strength in interactions.items():
            i1, i2 = hormones.index(h1), hormones.index(h2)
            matrix[i1, i2] = strength
            
        return matrix

    def update(self, dt: float):
        """Update hormone levels based on time passing and interactions"""
        # Update internal clock (24-hour cycle)
        self.internal_clock = (self.internal_clock + dt/3600) % 24
        
        # Apply circadian rhythm effects
        self._apply_circadian_effects()
        
        # Process hormone interactions
        self._process_hormone_interactions(dt)
        
        # Apply natural decay and homeostatic regulation
        self._apply_homeostasis(dt)

    def _apply_circadian_effects(self):
        """Apply time-of-day effects on hormones"""
        hour = self.internal_clock
        
        # Melatonin peaks at night
        night_factor = np.sin(((hour - 20) % 24) * np.pi / 12)
        self.hormone_states['melatonin'] += max(0, night_factor) * 0.1
        
        # Cortisol peaks in morning
        morning_factor = np.sin(((hour - 8) % 24) * np.pi / 12)
        self.hormone_states['cortisol'] += max(0, morning_factor) * 0.1
        
        # Serotonin affected by daylight
        day_factor = np.sin(((hour - 12) % 24) * np.pi / 12)
        self.hormone_states['serotonin'] += max(0, day_factor) * 0.05

    def _process_hormone_interactions(self, dt: float):
        """Process how hormones affect each other"""
        hormones = list(self.hormone_states.keys())
        
        # Get current levels for interaction calculation
        current_levels = np.array([
            np.mean(self.hormone_states[h]) for h in hormones
        ])
        
        # Calculate interaction effects
        interaction_effects = self.interaction_matrix @ current_levels
        
        # Apply effects with time scaling
        for i, hormone in enumerate(hormones):
            effect = interaction_effects[i] * dt
            self.hormone_states[hormone] += effect
            
        # Ensure bounds
        for hormone in self.hormone_states:
            self.hormone_states[hormone] = np.clip(
                self.hormone_states[hormone], 0, 1)

    def _apply_homeostasis(self, dt: float):
        """Apply natural decay and homeostatic regulation"""
        for hormone, target in vars(self.homeostatic_targets).items():
            if hormone in self.hormone_states:
                current = self.hormone_states[hormone]
                decay = self.decay_rates[hormone] * dt
                
                # Calculate pull toward homeostatic target
                delta = (target - np.mean(current)) * decay
                
                # Apply with some randomness
                self.hormone_states[hormone] += delta * np.random.normal(1, 0.1, current.shape)
                self.hormone_states[hormone] = np.clip(self.hormone_states[hormone], 0, 1)

    def trigger_response(self, event_type: str, intensity: float):
        """Trigger hormone response to various events"""
        responses = {
            'pleasure': {'dopamine': 0.5, 'serotonin': 0.2, 'endorphins': 0.3},
            'stress': {'cortisol': 0.6, 'adrenaline': 0.4, 'serotonin': -0.2},
            'social': {'oxytocin': 0.4, 'vasopressin': 0.3, 'dopamine': 0.2},
            'pain': {'endorphins': 0.5, 'cortisol': 0.3, 'adrenaline': 0.4},
            'fear': {'adrenaline': 0.7, 'cortisol': 0.5, 'oxytocin': -0.2},
            'achievement': {'dopamine': 0.6, 'serotonin': 0.3, 'endorphins': 0.2}
        }
        
        if event_type in responses:
            for hormone, change in responses[event_type].items():
                # Apply change with intensity scaling and randomness
                delta = change * intensity * np.random.normal(1, 0.1, 128)
                self.hormone_states[hormone] += delta
                self.hormone_states[hormone] = np.clip(
                    self.hormone_states[hormone], 0, 1)

    def get_hormone_profile(self) -> HormoneProfile:
        """Get current hormone levels as a profile"""
        return HormoneProfile(
            **{h: float(np.mean(self.hormone_states[h])) 
               for h in self.hormone_states}
        )

    def get_hormone_vector(self) -> np.ndarray:
        """Get complete hormone state as a single vector"""
        return np.concatenate(list(self.hormone_states.values())).astype(np.float32)

    def get_emotional_influence(self) -> Dict[str, float]:
        """Calculate hormone influence on emotional states"""
        profile = self.get_hormone_profile()
        
        return {
            'happiness': (profile.dopamine * 0.4 + 
                        profile.serotonin * 0.4 +
                        profile.endorphins * 0.2),
            'anxiety': (profile.cortisol * 0.5 +
                       profile.adrenaline * 0.3 -
                       profile.serotonin * 0.2),
            'trust': (profile.oxytocin * 0.6 +
                     profile.vasopressin * 0.4),
            'alertness': (profile.adrenaline * 0.4 +
                         profile.cortisol * 0.3 -
                         profile.melatonin * 0.3),
            'calmness': (profile.serotonin * 0.4 +
                        profile.melatonin * 0.3 -
                        profile.cortisol * 0.3)


                        import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import time
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnergyState:
    cognitive_energy: float = 1.0  # Mental processing power
    physical_energy: float = 1.0   # Physical action capability
    recovery_rate: float = 1.0     # How fast energy replenishes
    stress_level: float = 0.0      # Affects energy consumption

@dataclass
class ResourceState:
    memory_allocation: float = 0.0  # Neural resource usage
    processing_load: float = 0.0    # Computational load
    attention_resources: float = 1.0 # Available attention
    learning_capacity: float = 1.0   # Current ability to learn

class Metabolism:
    """
    Manages energy distribution,this is a more advanced fatigue management than original.need 
     to add decresaseexhaustion or maturity scale
       resource allocation, and system maintenance.
    Simulates biological metabolism for cognitive and physical resources.
    """
    def __init__(self):
        # Core energy pools
        self.base_energy = 1.0
        self.max_energy = 1.0
        self.current_energy = EnergyState()
        
        # Resource tracking
        self.resources = ResourceState()
        
        # Activity costs (energy per second)
        self.energy_costs = {
            'thinking': 0.02,
            'learning': 0.05,
            'physical_action': 0.03,
            'emotional_processing': 0.02,
            'memory_access': 0.01,
            'dream_state': 0.01
        }
        
        # Performance curves (how performance scales with energy)
        self.performance_curves = {
            'cognitive': lambda e: 1.0 - 0.8 * np.exp(-2.0 * e),
            'physical': lambda e: 1.0 - 0.9 * np.exp(-2.5 * e),
            'emotional': lambda e: 1.0 - 0.7 * np.exp(-1.8 * e)
        }
        
        # System state
        self.last_update = time.time()
        self.total_uptime = 0.0
        self.maintenance_needed = False
        self.dream_cycles = 0
        
        # Adaptive parameters
        self.adaptation_rate = 0.1
        self.stress_tolerance = 0.7
        self.recovery_bonus = 1.0
        
        # Resource pools (for different types of processing)
        self.resource_pools = {
            'short_term_memory': 1.0,
            'working_memory': 1.0,
            'attention_focus': 1.0,
            'emotional_capacity': 1.0,
            'learning_buffer': 1.0,
            'prediction_capacity': 1.0,
        }
        
        # Track burned calories (as metaphor for processing cost)
        self.calories_burned = 0.0
        self.calorie_history = []

    def update(self, dt: float):
        """Update metabolic state"""
        self.total_uptime += dt
        
        # Natural energy recovery
        self._process_energy_recovery(dt)
        
        # Resource reallocation
        self._manage_resources(dt)
        
        # Check for maintenance needs
        self._check_maintenance_status()
        
        # Update adaptation parameters
        self._adapt_parameters(dt)
        
        # Track calorie burn
        self._update_calorie_tracking(dt)

    def _process_energy_recovery(self, dt: float):
        """Handle natural energy recovery and limits"""
        # Base recovery rate affected by stress
        effective_recovery = self.current_energy.recovery_rate * \
                           (1.0 - 0.5 * self.current_energy.stress_level)
        
        # Apply recovery
        self.current_energy.cognitive_energy = min(
            self.max_energy,
            self.current_energy.cognitive_energy + effective_recovery * dt * 0.1
        )
        self.current_energy.physical_energy = min(
            self.max_energy,
            self.current_energy.physical_energy + effective_recovery * dt * 0.08
        )

    def _manage_resources(self, dt: float):
        """Manage and reallocate resources based on needs"""
        # Decay resource usage
        self.resources.memory_allocation *= 0.95
        self.resources.processing_load *= 0.90
        
        # Recover attention resources
        self.resources.attention_resources = min(
            1.0,
            self.resources.attention_resources + dt * 0.1
        )
        
        # Update learning capacity based on energy and stress
        self.resources.learning_capacity = self.performance_curves['cognitive'](
            self.current_energy.cognitive_energy
        ) * (1.0 - 0.5 * self.current_energy.stress_level)

    def _check_maintenance_status(self):
        """Check if system needs maintenance (like sleep)"""
        # Trigger maintenance need if:
        # - Energy too low
        # - Too much uptime
        # - Resource depletion
        energy_low = min(self.current_energy.cognitive_energy,
                        self.current_energy.physical_energy) < 0.3
        long_uptime = self.total_uptime > 16 * 3600  # 16 hours
        resources_low = min(self.resource_pools.values()) < 0.3
        
        self.maintenance_needed = energy_low or long_uptime or resources_low

    def _adapt_parameters(self, dt: float):
        """Adapt metabolic parameters based on usage patterns"""
        # Increase stress tolerance with exposure
        if self.current_energy.stress_level > self.stress_tolerance:
            self.stress_tolerance = min(
                0.9,
                self.stress_tolerance + dt * self.adaptation_rate
            )
        
        # Adjust recovery bonus based on maintenance cycles
        if self.dream_cycles > 0:
            self.recovery_bonus = min(
                2.0,
                self.recovery_bonus + 0.1 * self.dream_cycles
            )

    def _update_calorie_tracking(self, dt: float):
        """Track calorie burn as metaphor for processing cost"""
        # Base metabolic rate
        base_burn = 0.1 * dt  # 0.1 calories per second base rate
        
        # Additional costs based on activity
        cognitive_cost = (1.0 - self.current_energy.cognitive_energy) * 0.2 * dt
        physical_cost = (1.0 - self.current_energy.physical_energy) * 0.3 * dt
        stress_cost = self.current_energy.stress_level * 0.1 * dt
        
        total_burn = base_burn + cognitive_cost + physical_cost + stress_cost
        self.calories_burned += total_burn
        
        # Keep history for analysis
        self.calorie_history.append((time.time(), total_burn))
        if len(self.calorie_history) > 1000:
            self.calorie_history.pop(0)

    def consume_energy(self, activity: str, intensity: float = 1.0) -> bool:
        """Attempt to consume energy for an activity"""
        if activity not in self.energy_costs:
            return False
            
        cost = self.energy_costs[activity] * intensity
        
        # Check if we have enough energy
        if activity in ['thinking', 'learning', 'memory_access']:
            if self.current_energy.cognitive_energy < cost:
                return False
            self.current_energy.cognitive_energy -= cost
        else:
            if self.current_energy.physical_energy < cost:
                return False
            self.current_energy.physical_energy -= cost
            
        return True

    def allocate_resources(self, resource_type: str, amount: float) -> bool:
        """Attempt to allocate resources of a specific type"""
        if resource_type not in self.resource_pools:
            return False
            
        if self.resource_pools[resource_type] < amount:
            return False
            
        self.resource_pools[resource_type] -= amount
        return True

    def dream_cycle_completed(self):
        """Called when a dream cycle completes"""
        self.dream_cycles += 1
        self.maintenance_needed = False
        self.total_uptime = 0.0  # Reset uptime
        
        # Boost recovery and clear stress
        self.current_energy.recovery_rate = self.recovery_bonus
        self.current_energy.stress_level *= 0.5
        
        # Replenish resource pools
        for pool in self.resource_pools:
            self.resource_pools[pool] = min(1.0, self.resource_pools[pool] + 0.3)

    def get_performance_multiplier(self, activity_type: str) -> float:
        """Get current performance level for an activity type"""
        if activity_type in self.performance_curves:
            energy = (self.current_energy.cognitive_energy 
                     if activity_type == 'cognitive'
                     else self.current_energy.physical_energy)
            return self.performance_curves[activity_type](energy)
        return 1.0

    def get_status_report(self) -> Dict[str, float]:
        """Get current metabolic status"""
        return {
            'cognitive_energy': self.current_energy.cognitive_energy,
            'physical_energy': self.current_energy.physical_energy,
            'stress_level': self.current_energy.stress_level,
            'recovery_rate': self.current_energy.recovery_rate,
            'maintenance_needed': float(self.maintenance_needed),
            'calories_burned': self.calories_burned,
            'average_burn_rate': np.mean([b for _, b in self.calorie_history[-100:]])
        }
