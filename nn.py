# nn.py

import numpy as np
from typing import List, Tuple, Any, Dict, Optional
import logging
import time

# ==============================================================================
# SECTION 1: THE NEURAL FOUNDATION
# A from-scratch, pure NumPy replacement for a neural network library,
# including explicit backpropagation for custom optimization.
# ==============================================================================

class Layer:
    """
    The abstract base class for all neural network layers.
    Defines the required forward and backward pass methods.
    """
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Calculates the output of the layer for a given input.
        """
        raise NotImplementedError

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Calculates the gradient to be passed to the previous layer and
        updates internal parameters (weights/biases) if they exist.
        """
        raise NotImplementedError

    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        """Allows layers to be called directly like a function."""
        return self.forward(input_data)

    def get_trainable_params(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Returns a list of (parameter, gradient) tuples for this layer."""
        return [] # Default for layers without trainable params

class Linear(Layer):
    """
    A dense, fully connected neural layer with biological properties.
    This replacement keeps the original behavior but stores the primary
    arrays in float16 as requested (note: some ops may upcast temporarily).
    """
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        # He initialization: sqrt(2 / fan_in)
        scale = np.sqrt(2.0 / float(input_size)).astype(np.float16) if isinstance(input_size, int) else np.float16(np.sqrt(2.0 / float(input_size)))
        self.weights = (np.random.randn(input_size, output_size).astype(np.float16) * scale)
        self.biases = np.zeros((1, output_size), dtype=np.float16)

        # Plasticity factors
        self.plasticity_rate = np.float16(0.01)
        self.plasticity_decay = np.float16(0.999)
        self.plastic_changes = np.zeros_like(self.weights, dtype=np.float16)

        # Neural fatigue
        self.fatigue = np.zeros(output_size, dtype=np.float16)
        self.fatigue_rate = np.float16(0.1)
        self.recovery_rate = np.float16(0.05)
        self.fatigue_threshold = np.float16(0.7)

        # Homeostatic scaling
        self.target_activity = np.float16(0.5)
        self.scaling_rate = np.float16(0.001)
        self.activity_history = np.ones(output_size, dtype=np.float16) * np.float16(0.5)

        # Memory bounds to prevent leaks
        self.max_plastic_magnitude = np.float16(1.0)
        self.plastic_cleanup_threshold = np.float16(0.001)

        # Gradient storage
        self.weights_gradient = None
        self.biases_gradient = None

        # Training mode flag
        self.training = True

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        # Accept inputs, store as float16
        self.input = input_data.astype(np.float16)

        # Linear transform (dot may upcast internally)
        raw_output = np.dot(self.input, self.weights) + self.biases

        # Fatigue modulation
        fatigue_mask = np.float16(1.0) - np.clip(self.fatigue - self.fatigue_threshold, np.float16(0.0), np.float16(1.0))
        modulated_output = raw_output * fatigue_mask

        # Update fatigue
        self.fatigue += self.fatigue_rate * (np.abs(modulated_output) > np.float16(0.1)).mean(axis=0)
        self.fatigue *= (np.float16(1.0) - self.recovery_rate)

        # Hebbian plasticity (simple correlation-based)
        if self.training:
            correlation = np.dot(self.input.T, modulated_output) / max(1, self.input.shape[0])
            self.plastic_changes = (self.plastic_changes * self.plasticity_decay) + (correlation * self.plasticity_rate)
            self.plastic_changes = np.clip(self.plastic_changes, -self.max_plastic_magnitude, self.max_plastic_magnitude)
            small_changes_mask = np.abs(self.plastic_changes) < self.plastic_cleanup_threshold
            self.plastic_changes[small_changes_mask] *= np.float16(0.1)
            effective_weights = self.weights + self.plastic_changes
        else:
            effective_weights = self.weights

        # Final output
        self.output = np.dot(self.input, effective_weights) + self.biases

        # Activity history and homeostatic scaling
        current_activity = (self.output > 0).mean(axis=0).astype(np.float16)
        self.activity_history = (self.activity_history * np.float16(0.9)) + (current_activity * np.float16(0.1))
        activity_scale = (np.float16(1.0) + (self.scaling_rate * (self.target_activity - self.activity_history)))
        self.output = (self.output * activity_scale).astype(np.float16)

        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        # Scale gradient based on fatigue (upcast for numerical ops where needed)
        fatigue_mask = (np.float16(1.0) - np.clip(self.fatigue - self.fatigue_threshold, np.float16(0.0), np.float16(1.0))).astype(np.float16)
        scaled_gradient = (output_gradient * fatigue_mask).astype(np.float16)

        effective_weights = (self.weights + self.plastic_changes).astype(np.float16)
        self.weights_gradient = np.dot(self.input.T, scaled_gradient)
        self.biases_gradient = np.sum(scaled_gradient, axis=0, keepdims=True)

        activity_error = (self.activity_history - self.target_activity).astype(np.float16)
        homeostatic_gradient = (self.scaling_rate * activity_error * np.sign(self.output)).astype(np.float16)
        if self.weights_gradient.shape[0] == 1 and homeostatic_gradient.ndim == 1:
            self.weights_gradient += homeostatic_gradient.reshape(1, -1)
        else:
            try:
                self.weights_gradient += homeostatic_gradient.reshape(1, -1)
            except Exception:
                pass

        input_gradient = np.dot(scaled_gradient, effective_weights.T).astype(np.float16)
        return input_gradient

    def get_trainable_params(self) -> List[Tuple[np.ndarray, Any]]:
        """Returns list of (parameter_array, gradient_array) tuples."""
        return [(self.weights, 'weights_gradient'), (self.biases, 'biases_gradient')]

    def train(self):
        """Set the layer to training mode."""
        self.training = True

    def eval(self):
        """Set the layer to evaluation mode."""
        self.training = False

class Tanh(Layer):
    """A hyperbolic tangent activation function."""
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        self.output = np.tanh(self.input)
        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        # Derivative of tanh(x) is 1 - tanh(x)^2
        return output_gradient * (1 - self.output**2)

class Sigmoid(Layer):
    """A sigmoid activation function, squashing values between 0 and 1."""
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        self.output = 1 / (1 + np.exp(-self.input))
        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        # Derivative of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x))
        return output_gradient * (self.output * (1 - self.output))

class ReLU(Layer):
    """A Rectified Linear Unit activation function."""
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        self.output = np.maximum(0, self.input)
        return self.output
        
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        # Derivative of ReLU is 1 for positive inputs and 0 otherwise.
        relu_mask = (self.input > 0).astype(np.float16)
        return output_gradient * relu_mask

class GELU(Layer):
    """A Gaussian Error Linear Unit, a smoother version of ReLU."""
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        # Standard GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        self.output = 0.5 * self.input * (1 + np.tanh(np.sqrt(2 / np.pi) * (self.input + 0.044715 * self.input**3)))
        return self.output
        
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        # Approximate derivative for the GELU function
        x = self.input
        tanh_arg = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
        tanh_val = np.tanh(tanh_arg)
        sech_squared_val = 1 - tanh_val**2 
        derivative_inner_term = np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)
        
        derivative = 0.5 * (1 + tanh_val) + 0.5 * x * sech_squared_val * derivative_inner_term
        
        return output_gradient * derivative

class LayerNorm(Layer):
    """
    Applies Layer Normalization over the last dimension of the input.
    Helps stabilize learning in deep networks.
    """
    def __init__(self, features: int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        # Gamma (scale) and Beta (shift) are learnable parameters
        self.gamma = np.ones((1, features), dtype=np.float16)
        self.beta = np.zeros((1, features), dtype=np.float16)

        # Gradients for gamma and beta
        self.gamma_gradient = None
        self.beta_gradient = None

        # Internal variables saved for backward pass
        self.mean = None
        self.variance = None
        self.normalized_input = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data.astype(np.float16)
        
        # Calculate mean and variance over the last dimension (features)
        self.mean = np.mean(self.input, axis=-1, keepdims=True)
        self.variance = np.var(self.input, axis=-1, keepdims=True)
        
        # Normalize the input
        self.normalized_input = (self.input - self.mean) / np.sqrt(self.variance + self.epsilon)
        
        # Apply learnable scaling (gamma) and shifting (beta)
        self.output = self.gamma * self.normalized_input + self.beta
        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        # Gradients for beta and gamma
        self.beta_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        self.gamma_gradient = np.sum(output_gradient * self.normalized_input, axis=0, keepdims=True)

        # Gradient with respect to normalized input (dL/d(normalized_input))
        d_normalized_input = output_gradient * self.gamma

        # Gradient with respect to variance (dL/d(variance))
        d_variance = np.sum(d_normalized_input * (self.input - self.mean) * -0.5 * np.power(self.variance + self.epsilon, -1.5), axis=-1, keepdims=True)

        # Gradient with respect to mean (dL/d(mean))
        # d_mean is sum of gradients of (input-mean) and variance wrt mean
        d_mean = np.sum(d_normalized_input * (-1 / np.sqrt(self.variance + self.epsilon)), axis=-1, keepdims=True) + \
                 d_variance * (-2 * (self.input - self.mean)).mean(axis=-1, keepdims=True) # mean across features

        # Gradient with respect to input data (dL/d(input_data))
        # This part assumes input_data is 2D (batch, features)
        input_gradient = d_normalized_input * (1 / np.sqrt(self.variance + self.epsilon)) + \
                         d_variance * (2 * (self.input - self.mean) / self.input.shape[-1]) + \
                         d_mean / self.input.shape[-1]
        
        return input_gradient

    def get_trainable_params(self) -> List[Tuple[np.ndarray, Any]]:
        """Returns list of (parameter_array, gradient_string) tuples for LayerNorm."""
        return [(self.gamma, 'gamma_gradient'), (self.beta, 'beta_gradient')]

class Sequential:
    """
    A container to chain layers together and handle the forward and backward passes.
    Manages collection of trainable parameters and their gradients from contained layers.
    """
    def __init__(self, *layers: Layer):
        self.layers = layers

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        output = input_data
        for layer in self.layers:
            output = layer(output)
        return output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """Propagates the gradient backwards through all layers in reverse order."""
        grad = output_gradient
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        """Allows Sequential model to be called directly like a function."""
        return self.forward(input_data)

    def get_trainable_params(self) -> List[Tuple[np.ndarray, Any, Layer]]:
        """
        Returns a list of (parameter_array, gradient_string, layer_instance) tuples
        for all trainable parameters in the network.
        """
        params_list = []
        for layer in self.layers:
            layer_params = layer.get_trainable_params()
            for param_array, grad_name in layer_params:
                params_list.append((param_array, grad_name, layer))
        return params_list

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calculates the Mean Squared Error loss."""
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)
    return np.mean(np.square(y_pred - y_true)).astype(np.float32)

#mean squared error loss derivative
def mse_loss_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calculates the derivative of the Mean Squared Error loss function."""
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)
    return 2 * (y_pred - y_true) / np.size(y_true)


# ==============================================================================
# SECTION 2: OPTIMIZERS
# ==============================================================================

class AdamW:
    """
    AdamW optimizer from scratch.
    Combines adaptive learning rates (Adam) with decoupled weight decay (W).
    """
    def __init__(self, networks: List[Sequential], learning_rate: float = 1e-3, 
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8,
                 weight_decay: float = 0.01):
        self.lr = np.float32(learning_rate)
        self.beta1 = np.float32(beta1)
        self.beta2 = np.float32(beta2)
        self.epsilon = np.float32(epsilon)
        self.weight_decay = np.float32(weight_decay) # L2 regularization

        self.networks = networks
        
        # Initialize momentum (m) and velocity (v) for each parameter
        # Stored as dictionaries, mapping (network_id, layer_id, param_type) to state
        self.m = {} 
        self.v = {}
        self.t = 0 # Timestep

    def step(self):
        """Performs a single optimization step."""
        self.t += 1
        
        # Bias correction terms
        lr_t = self.lr * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)

        for net_idx, net in enumerate(self.networks):
            # Iterate through all trainable parameters in the network
            for param_array, grad_name, layer_instance in net.get_trainable_params():
                # Get the actual gradient using getattr
                grad_array = getattr(layer_instance, grad_name)

                if grad_array is None: # Skip if gradient was not computed (e.g. for non-active path)
                    continue

                # Create a unique key for this parameter (network instance, layer instance, parameter type)
                param_key = (id(net), id(layer_instance), grad_name)

                # Initialize m and v for this parameter if not already done
                if param_key not in self.m:
                    self.m[param_key] = np.zeros_like(param_array, dtype=np.float32)
                    self.v[param_key] = np.zeros_like(param_array, dtype=np.float32)

                # Update biased first and second moment estimates
                self.m[param_key] = self.beta1 * self.m[param_key] + (1 - self.beta1) * grad_array
                self.v[param_key] = self.beta2 * self.v[param_key] + (1 - self.beta2) * (grad_array**2)
                
                # Update parameters (decoupled weight decay for AdamW)
                # param = param - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)
                
                # Bias-corrected first and second moment estimates are implicitly in lr_t
                # But for decoupled weight decay, apply weight_decay separately to the parameter itself.
                update = lr_t * self.m[param_key] / (np.sqrt(self.v[param_key]) + self.epsilon)
                
                # Apply update to the parameter
                param_array -= (update + self.lr * self.weight_decay * param_array)


class JustinJOptimizer:
    """
    An optimizer designed to develop self-awareness and agency through:
    1. Audio-motor coupling (connecting voice output to self-perception)
    2. Intentional control development
    3. Goal-oriented behavior emergence
    4. Multi-timescale learning
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
                 weight_decay: float = 0.01):

        # Initialize base AdamW optimizer
        self.adamw = AdamW(networks,
                          learning_rate=base_lr,
                          beta1=beta1,
                          beta2=beta2,
                          epsilon=epsilon,
                          weight_decay=weight_decay)

        self.networks = networks
        self.base_lr = np.float32(base_lr)
        self.vocal_feedback_weight = np.float32(vocal_feedback_weight)
        self.agency_growth_rate = np.float32(agency_growth_rate)
        self.control_precision = np.float32(control_precision)

        # Agency potential tracking
        self.p = {}  # Agency potential (how much a parameter influences agency)

        # Agency development tracking
        self.voice_control_confidence = 0.0  # Grows as voice control improves
        self.intention_alignment = 0.0       # Grows as actions match intentions
        self.feedback_recognition = 0.0      # Grows as system recognizes own outputs

        # Multi-timescale learning rates
        self.fast_lr = self.base_lr * 2.0   # For immediate feedback
        self.slow_lr = self.base_lr * 0.5   # For pattern consolidation

        # Temporal context windows
        self.short_term_memory = []  # Recent input-output pairs
        self.pattern_memory = []     # Successful control patterns

        self.t = 0  # Timestep counter
        
        # Gradient clipping to prevent exploding gradients
        self.max_grad_norm = 1.0  # Maximum gradient norm
        self.clip_gradients = True  # Enable gradient clipping

        logger = logging.getLogger(__name__)
        logger.info("JustinJOptimizer initialized with focus on developing self-awareness and control")

    def update_metrics(self,
                      vocal_output: np.ndarray,
                      audio_feedback: np.ndarray,
                      intended_output: Optional[np.ndarray] = None,
                      reward_signal: Optional[float] = None):
        """
        Update internal metrics based on system feedback
        """
        # Calculate voice feedback alignment
        if vocal_output is not None and audio_feedback is not None:
            feedback_match = self._calculate_feedback_match(vocal_output, audio_feedback)
            self.feedback_recognition = (self.feedback_recognition * 0.95 +
                                      feedback_match * 0.05)

        # Update intention alignment if provided
        if intended_output is not None and vocal_output is not None:
            control_accuracy = self._calculate_control_accuracy(intended_output, vocal_output)
            self.intention_alignment = (self.intention_alignment * 0.9 +
                                     control_accuracy * 0.1)

        # Update voice control confidence
        self.voice_control_confidence = (self.feedback_recognition * 0.6 +
                                       self.intention_alignment * 0.4)

        # Store context for pattern learning
        self._update_temporal_context(vocal_output, audio_feedback, intended_output, reward_signal)

    def _calculate_feedback_match(self, vocal_output: np.ndarray, audio_feedback: np.ndarray) -> float:
        """Calculate how well audio feedback matches expected output"""
        # Normalize both signals
        vocal_norm = vocal_output / (np.max(np.abs(vocal_output)) + 1e-8)
        feedback_norm = audio_feedback / (np.max(np.abs(audio_feedback)) + 1e-8)

        # Calculate correlation
        correlation = np.corrcoef(vocal_norm.flatten(), feedback_norm.flatten())[0,1]
        return max(0, correlation)  # Only positive correlation matters for recognition

    def _calculate_control_accuracy(self, intended: np.ndarray, actual: np.ndarray) -> float:
        """Calculate how well actual output matches intentions"""
        error = np.mean(np.square(intended - actual))
        return np.exp(-error / self.control_precision)

    def _update_temporal_context(self, vocal_output, audio_feedback, intended_output, reward):
        """Update temporal context windows for pattern learning"""
        context = {
            'time': time.time(),
            'vocal_output': vocal_output,
            'audio_feedback': audio_feedback,
            'intended_output': intended_output,
            'reward': reward
        }

        # Update short-term memory
        self.short_term_memory.append(context)
        if len(self.short_term_memory) > 100:  # Keep last 100 timesteps
            self.short_term_memory.pop(0)

        # Update pattern memory if this was a successful interaction
        if reward is not None and reward > 0.7:  # High reward threshold
            self.pattern_memory.append(context)
            if len(self.pattern_memory) > 1000:  # Keep last 1000 successful patterns
                self.pattern_memory.pop(0)

    def step(self):
        """Perform an optimization step combining AdamW with agency development"""
        self.t += 1

        # Calculate agency-based learning rate modulation
        agency_factor = np.sqrt(self.voice_control_confidence + 0.1)

        # Temporarily modify AdamW's learning rate based on agency development
        original_lr = self.adamw.lr
        self.adamw.lr = original_lr * agency_factor

        for net_idx, net in enumerate(self.networks):
            for param_array, grad_name, layer_instance in net.get_trainable_params():
                grad_array = getattr(layer_instance, grad_name)
                if grad_array is None:
                    continue

                param_key = (id(net), id(layer_instance), grad_name)

                # Initialize agency potential if needed
                if param_key not in self.p:
                    self.p[param_key] = np.ones_like(param_array, dtype=np.float32)

                # Update agency potential
                if self.voice_control_confidence > self.p[param_key].mean():
                    # Parameter helped improve agency - strengthen it
                    self.p[param_key] *= (1.0 + self.agency_growth_rate)
                else:
                    # Parameter might be interfering - weaken it slightly
                    self.p[param_key] *= (1.0 - self.agency_growth_rate * 0.1)

                # Modify gradients based on agency potential and temporal context
                if len(self.pattern_memory) > 0:
                    # More emphasis on established patterns
                    grad_array *= np.sqrt(self.p[param_key]) * 1.2
                else:
                    # More emphasis on exploration
                    grad_array *= np.sqrt(self.p[param_key]) * 0.8

                # Update the gradient in the layer instance for AdamW
                setattr(layer_instance, grad_name, grad_array)

        # Apply gradient clipping before AdamW step
        if self.clip_gradients:
            self._clip_gradients()

        # Let AdamW perform its optimization step
        self.adamw.step()

        # Restore original learning rate
        self.adamw.lr = original_lr

        # Log progress periodically
        if self.t % 100 == 0:
            logger = logging.getLogger(__name__)
            logger.info(f"Agency Metrics - Voice Control: {self.voice_control_confidence:.3f}, "
                       f"Intention Alignment: {self.intention_alignment:.3f}, "
                       f"Feedback Recognition: {self.feedback_recognition:.3f}")

    def get_agency_metrics(self) -> Dict[str, float]:
        """Return current agency development metrics"""
        return {
            'voice_control_confidence': float(self.voice_control_confidence),
            'intention_alignment': float(self.intention_alignment),
            'feedback_recognition': float(self.feedback_recognition),
            'pattern_memory_size': len(self.pattern_memory)
        }

    def _clip_gradients(self):
        """Clip gradients to prevent exploding gradients"""
        total_norm = 0.0
        
        # Calculate total gradient norm across all parameters
        for net in self.networks:
            for param_array, grad_name, layer_instance in net.get_trainable_params():
                grad_array = getattr(layer_instance, grad_name)
                if grad_array is not None:
                    param_norm = np.linalg.norm(grad_array)
                    total_norm += param_norm ** 2
        
        total_norm = np.sqrt(total_norm)
        
        # Clip gradients if norm exceeds threshold
        if total_norm > self.max_grad_norm:
            clip_coef = self.max_grad_norm / (total_norm + 1e-6)
            
            # Apply clipping to all gradients
            for net in self.networks:
                for param_array, grad_name, layer_instance in net.get_trainable_params():
                    grad_array = getattr(layer_instance, grad_name)
                    if grad_array is not None:
                        clipped_grad = grad_array * clip_coef
                        setattr(layer_instance, grad_name, clipped_grad)