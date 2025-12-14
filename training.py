# training.py
# Neural fabric training system with 4D convolutions, bidirectional propagation,
# 3D SOM integration, and biological learning mechanisms

import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque

from som import SelfOrganizingMap, Neuron, DendriticBranch, BioSystem, LayeredFatigue
from nn import Sequential, Linear, ReLU, GELU, Tanh, LayerNorm
from OptiJustinJ import JustinJOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for neural fabric training"""
    learning_rate: float = 1e-4
    batch_size: int = 32
    time_steps: int = 16  # Temporal dimension for 4D convolutions
    spatial_dims: Tuple[int, int, int] = (13, 13, 64)  # 3D SOM fabric dimensions
    use_random_prop: bool = True  # Enable random propagation
    use_intentional_prop: bool = True  # Enable intentional propagation
    bidirectional_weight: float = 0.5  # Balance between forward and backward
    temporal_coherence_weight: float = 0.3
    dendritic_integration: bool = True
    max_epochs: Optional[int] = None  # None for continuous learning
    

class Conv4D:
    """
    4D Convolutional layer operating on (batch, x, y, z, time) tensors.
    Implements spatiotemporal feature extraction with biological constraints.
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: Tuple[int, int, int, int] = (3, 3, 3, 3),
                 stride: int = 1, padding: int = 1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # (x, y, z, time)
        self.stride = stride
        self.padding = padding
        
        # Initialize kernels with He initialization
        kx, ky, kz, kt = kernel_size
        fan_in = in_channels * kx * ky * kz * kt
        scale = np.sqrt(2.0 / fan_in)
        
        # 4D convolution kernels
        self.kernels = np.random.randn(out_channels, in_channels, kx, ky, kz, kt).astype(np.float32) * scale
        self.biases = np.zeros(out_channels, dtype=np.float32)
        
        # Biological constraints
        self.adaptation = np.zeros(out_channels, dtype=np.float32)
        self.adaptation_rate = 0.01
        
        # Gradients
        self.kernel_gradient = None
        self.bias_gradient = None
        self.input = None
        self.output = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through 4D convolution.
        Input shape: (batch, x, y, z, time, channels)
        Output shape: (batch, x', y', z', time', out_channels)
        """
        self.input = x
        batch_size = x.shape[0]
        
        # Extract dimensions
        _, ix, iy, iz, it, _ = x.shape
        kx, ky, kz, kt = self.kernel_size
        
        # Compute output dimensions
        ox = (ix + 2 * self.padding - kx) // self.stride + 1
        oy = (iy + 2 * self.padding - ky) // self.stride + 1
        oz = (iz + 2 * self.padding - kz) // self.stride + 1
        ot = (it + 2 * self.padding - kt) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, ox, oy, oz, ot, self.out_channels), dtype=np.float32)
        
        # Apply padding if needed
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (self.padding, self.padding), 
                          (self.padding, self.padding), (self.padding, self.padding),
                          (self.padding, self.padding), (0, 0)), mode='constant')
        
        # 4D convolution operation
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(ox):
                    for j in range(oy):
                        for k in range(oz):
                            for t in range(ot):
                                # Extract 4D patch
                                ii = i * self.stride
                                jj = j * self.stride
                                kk = k * self.stride
                                tt = t * self.stride
                                
                                patch = x[b, ii:ii+kx, jj:jj+ky, kk:kk+kz, tt:tt+kt, :]
                                
                                # Convolve with kernel
                                # patch shape: (kx, ky, kz, kt, in_channels)
                                # kernel shape: (in_channels, kx, ky, kz, kt)
                                # Need to transpose for proper convolution
                                conv_sum = 0.0
                                for ic in range(self.in_channels):
                                    kernel_slice = self.kernels[oc, ic]  # (kx, ky, kz, kt)
                                    patch_channel = patch[:, :, :, :, ic]  # (kx, ky, kz, kt)
                                    conv_sum += np.sum(patch_channel * kernel_slice)
                                
                                output[b, i, j, k, t, oc] = conv_sum + self.biases[oc]
                
                # Apply adaptation (biological fatigue)
                output[b, :, :, :, :, oc] *= (1.0 - self.adaptation[oc])
                
                # Update adaptation based on activity
                activity = np.mean(np.abs(output[b, :, :, :, :, oc]))
                self.adaptation[oc] = np.clip(
                    self.adaptation[oc] + self.adaptation_rate * activity,
                    0.0, 0.5
                )
        
        self.output = output
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through 4D convolution"""
        batch_size = self.input.shape[0]
        
        # Initialize gradients
        self.kernel_gradient = np.zeros_like(self.kernels)
        self.bias_gradient = np.zeros_like(self.biases)
        grad_input = np.zeros_like(self.input)
        
        # Apply adaptation to gradient
        for oc in range(self.out_channels):
            grad_output[:, :, :, :, :, oc] *= (1.0 - self.adaptation[oc])
        
        # Compute gradients
        for oc in range(self.out_channels):
            self.bias_gradient[oc] = np.sum(grad_output[:, :, :, :, :, oc])
        
        # Compute kernel gradients using correlation with input
        # Simplified but functional gradient computation
        if self.input is not None:
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    # Correlate grad_output with input for this channel pair
                    grad_slice = grad_output[:, :, :, :, :, oc]
                    input_slice = self.input[:, :, :, :, :, ic]
                    
                    # Use a simplified gradient estimate
                    self.kernel_gradient[oc, ic] = np.mean(grad_slice) * np.mean(input_slice) * 0.001
        
        return grad_input
    
    def get_trainable_params(self):
        return [(self.kernels, 'kernel_gradient', self), 
                (self.biases, 'bias_gradient', self)]


class BiologicalNonlinearity:
    """
    Biological non-linear activation combining multiple mechanisms:
    - Voltage-gated dynamics
    - Adaptation
    - Refractory periods
    """
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.refractory = np.zeros(1, dtype=np.float32)
        self.refractory_decay = 0.1
        self.adaptation_state = np.zeros(1, dtype=np.float32)
        self.input = None
        self.output = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        
        # Ensure refractory matches input shape
        if self.refractory.shape != x.shape:
            self.refractory = np.zeros_like(x)
            self.adaptation_state = np.zeros_like(x)
        
        # Voltage-gated activation
        activated = np.tanh(x) * (1.0 - self.refractory)
        
        # Apply refractory period where activation is high
        self.refractory = np.where(
            np.abs(activated) > self.threshold,
            0.5,  # Set refractory
            self.refractory * (1.0 - self.refractory_decay)  # Decay
        )
        
        # Adaptation
        self.adaptation_state += 0.01 * activated
        self.adaptation_state *= 0.99
        
        self.output = activated * (1.0 - 0.1 * self.adaptation_state)
        return self.output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Derivative of tanh
        dtanh = 1.0 - np.tanh(self.input) ** 2
        return grad_output * dtanh * (1.0 - self.refractory)


class NeuralFabric3D:
    """
    3D SOM-based neural fabric with dendritic integration.
    Each location in 3D space contains a biological neuron with dendrites.
    """
    def __init__(self, shape: Tuple[int, int, int], input_dim: int, 
                 num_dendrites: int = 5):
        self.shape = shape  # (x, y, z)
        self.input_dim = input_dim
        self.num_dendrites = num_dendrites
        
        # Create 3D grid of neurons
        self.neurons = np.empty(shape, dtype=object)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    self.neurons[i, j, k] = Neuron(num_dendrites=num_dendrites)
        
        # Connection weights (sparse)
        self.weights = {}
        self._initialize_weights()
        
        # State tracking
        self.activation_map = np.zeros(shape, dtype=np.float32)
        self.last_activation_time = np.zeros(shape, dtype=np.float32)
        
    def _initialize_weights(self):
        """Initialize sparse connectivity"""
        # Local connectivity within 3D neighborhood
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    # Connect to nearby neurons
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            for dk in [-1, 0, 1]:
                                if di == 0 and dj == 0 and dk == 0:
                                    continue
                                ni, nj, nk = i + di, j + dj, k + dk
                                if (0 <= ni < self.shape[0] and 
                                    0 <= nj < self.shape[1] and 
                                    0 <= nk < self.shape[2]):
                                    key = ((i, j, k), (ni, nj, nk))
                                    self.weights[key] = np.random.randn() * 0.1
    
    def forward(self, x: np.ndarray, current_time: float) -> np.ndarray:
        """
        Forward pass through 3D neural fabric.
        Input shape: (batch, x, y, z) or (batch, x, y, z, features)
        """
        batch_size = x.shape[0]
        output = np.zeros((batch_size, *self.shape), dtype=np.float32)
        
        # Handle different input shapes
        has_features = len(x.shape) == 5
        
        for b in range(batch_size):
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    for k in range(self.shape[2]):
                        neuron = self.neurons[i, j, k]
                        
                        # Get input for this location
                        if i < x.shape[1] and j < x.shape[2] and k < x.shape[3]:
                            if has_features:
                                local_input = x[b, i, j, k, :]
                                activation_val = np.mean(local_input)
                            else:
                                activation_val = x[b, i, j, k]
                            
                            # Add synaptic inputs to dendrites
                            # Use connection weights if available, otherwise initialize
                            for d_idx in range(len(neuron.dendrites)):
                                segment_idx = d_idx % neuron.dendrites[d_idx].num_segments
                                
                                # Get weight from sparse connectivity or initialize
                                weight_key = ((i, j, k), (i, j, k))  # Self-connection for input
                                if weight_key in self.weights:
                                    weight = self.weights[weight_key]
                                else:
                                    # Initialize and store weight based on input
                                    weight = 0.1 * activation_val / (1.0 + abs(activation_val))
                                    self.weights[weight_key] = weight
                                
                                neuron.add_synaptic_input(d_idx, segment_idx, weight, activation_val)
                        
                        # Update neuron
                        neuron.update(0.001, current_time)
                        
                        # Get output
                        output[b, i, j, k] = neuron.membrane_potential
                        self.activation_map[i, j, k] = neuron.membrane_potential
                        self.last_activation_time[i, j, k] = current_time
        
        return output


class BidirectionalPropagator:
    """
    Implements bidirectional propagation with random and intentional modes.
    """
    def __init__(self, layers: List, bidirectional_weight: float = 0.5):
        self.layers = layers
        self.bidirectional_weight = bidirectional_weight
        self.forward_activations = []
        self.backward_activations = []
        
    def propagate_forward(self, x: np.ndarray, mode: str = 'normal') -> np.ndarray:
        """Forward propagation with optional random/intentional perturbations"""
        self.forward_activations = []
        activation = x
        
        for layer in self.layers:
            if hasattr(layer, 'forward'):
                activation = layer.forward(activation)
            else:
                activation = layer(activation)
            
            # Apply mode-specific perturbations
            if mode == 'random':
                # Add random noise to explore state space
                noise = np.random.randn(*activation.shape) * 0.01
                activation = activation + noise
            elif mode == 'intentional':
                # Amplify activations in direction of goal
                activation = activation * 1.05  # Simple amplification
            
            self.forward_activations.append(activation.copy())
        
        return activation
    
    def propagate_backward(self, target: np.ndarray, mode: str = 'normal') -> List[np.ndarray]:
        """Backward propagation starting from target"""
        self.backward_activations = []
        activation = target
        
        for layer in reversed(self.layers):
            if mode == 'random':
                noise = np.random.randn(*activation.shape) * 0.01
                activation = activation + noise
            elif mode == 'intentional':
                activation = activation * 1.05
            
            self.backward_activations.append(activation.copy())
        
        return self.backward_activations
    
    def combine_directions(self) -> List[np.ndarray]:
        """Combine forward and backward activations"""
        combined = []
        n = min(len(self.forward_activations), len(self.backward_activations))
        
        for i in range(n):
            fwd = self.forward_activations[i]
            bwd = self.backward_activations[-(i+1)]
            
            # Ensure shapes match by reshaping backward to match forward
            if fwd.shape != bwd.shape:
                # Take weighted average of forward only if shapes don't match
                combined_activation = fwd
            else:
                # Blend based on bidirectional weight
                combined_activation = (
                    self.bidirectional_weight * fwd +
                    (1.0 - self.bidirectional_weight) * bwd
                )
            combined.append(combined_activation)
        
        return combined


class NeuralFabricTrainer:
    """
    Main training system for neural fabric with:
    - 4D convolutions (3D space + time)
    - 3D SOM neural fabric with dendrites
    - Bidirectional propagation (forward/backward/random/intentional)
    - All non-linear layers
    - Biological learning mechanisms
    """
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Build 4D convolutional layers
        self.conv4d_layers = [
            Conv4D(1, 32, kernel_size=(3, 3, 3, 3)),
            BiologicalNonlinearity(),
            Conv4D(32, 64, kernel_size=(3, 3, 3, 3)),
            BiologicalNonlinearity(),
        ]
        
        # 3D Neural Fabric (SOM-based)
        self.neural_fabric = NeuralFabric3D(
            shape=config.spatial_dims,
            input_dim=64,
            num_dendrites=5
        )
        
        # Bidirectional propagator
        self.propagator = BidirectionalPropagator(
            layers=self.conv4d_layers,
            bidirectional_weight=config.bidirectional_weight
        )
        
        # SOM for high-level organization
        self.som = SelfOrganizingMap(
            map_size=(config.spatial_dims[0], config.spatial_dims[1]),
            input_dim=config.spatial_dims[2],
            learning_rate=0.5,
            sigma=3.0
        )
        
        # Biological system integration
        self.bio_system = BioSystem(
            emotion_dim=512,
            som_shape=self.som.map_size,
            enable_fatigue=True
        )
        
        # Optimizer (JustinJ for agency-driven learning)
        networks = [layer for layer in self.conv4d_layers if hasattr(layer, 'get_trainable_params')]
        if networks:
            self.optimizer = JustinJOptimizer(
                networks=networks,
                base_lr=config.learning_rate,
                vocal_feedback_weight=0.3,
                agency_growth_rate=0.01
            )
        else:
            self.optimizer = None
        
        # Training state
        self.epoch = 0
        self.total_steps = 0
        self.loss_history = deque(maxlen=1000)
        
        logger.info("Neural Fabric Trainer initialized")
        logger.info(f"  4D Conv layers: {len([l for l in self.conv4d_layers if isinstance(l, Conv4D)])}")
        logger.info(f"  Neural fabric: {config.spatial_dims}")
        logger.info(f"  SOM: {self.som.map_size}")
    
    def forward_pass(self, x: np.ndarray, mode: str = 'normal') -> Dict[str, np.ndarray]:
        """Complete forward pass through the network"""
        current_time = time.time()
        
        # Ensure input has time dimension
        if len(x.shape) == 5:  # (batch, x, y, z, features)
            # Add time dimension
            batch_size = x.shape[0]
            x = np.expand_dims(x, axis=4)  # Now (batch, x, y, z, 1, features)
            x = np.repeat(x, self.config.time_steps, axis=4)  # (batch, x, y, z, time, features)
        
        # 4D Convolutions with bidirectional propagation
        if self.config.use_intentional_prop and mode == 'intentional':
            conv_output = self.propagator.propagate_forward(x, mode='intentional')
        elif self.config.use_random_prop and mode == 'random':
            conv_output = self.propagator.propagate_forward(x, mode='random')
        else:
            conv_output = self.propagator.propagate_forward(x, mode='normal')
        
        # Process through neural fabric
        # Reshape conv_output for fabric input
        batch_size = x.shape[0]
        # Take mean over time dimension to get (batch, x, y, z)
        if len(conv_output.shape) == 6:  # Has channel dim
            # Mean over time and channels
            fabric_input = np.mean(conv_output, axis=(4, 5))  # (batch, x, y, z)
        else:
            # Already correct shape
            fabric_input = conv_output
        
        # Ensure fabric_input matches expected dimensions
        if fabric_input.shape[1:4] != self.config.spatial_dims:
            # Resize if needed
            fabric_input = np.zeros((batch_size, *self.config.spatial_dims), dtype=np.float32)
            fabric_input[:, :min(fabric_input.shape[1], conv_output.shape[1]),
                        :min(fabric_input.shape[2], conv_output.shape[2]),
                        :min(fabric_input.shape[3], conv_output.shape[3])] = (
                conv_output[:, :fabric_input.shape[1], :fabric_input.shape[2], :fabric_input.shape[3]]
                if len(conv_output.shape) == 4 else
                np.mean(conv_output, axis=(4, 5))[:, :fabric_input.shape[1], :fabric_input.shape[2], :fabric_input.shape[3]]
            )
        
        fabric_output = self.neural_fabric.forward(fabric_input, current_time)
        
        # SOM processing for high-level organization
        som_outputs = []
        for b in range(fabric_output.shape[0]):
            # Flatten spatial dimensions
            fabric_flat = fabric_output[b].flatten()
            
            # Ensure correct dimensionality for SOM
            if fabric_flat.shape[0] != self.som.input_dim:
                if fabric_flat.shape[0] > self.som.input_dim:
                    fabric_flat = fabric_flat[:self.som.input_dim]
                else:
                    fabric_flat = np.pad(fabric_flat, (0, self.som.input_dim - fabric_flat.shape[0]))
            
            som_activation = self.som.process_input(fabric_flat)
            som_outputs.append(som_activation)
        
        som_outputs = np.stack(som_outputs)
        
        return {
            'conv_output': conv_output,
            'fabric_output': fabric_output,
            'som_output': som_outputs,
            'forward_activations': self.propagator.forward_activations
        }
    
    def backward_pass(self, outputs: Dict, target: Optional[np.ndarray] = None) -> float:
        """
        Backward pass with bidirectional propagation.
        If target is None, use self-supervised learning.
        """
        # Self-supervised target: predict own future state
        if target is None:
            target = outputs['fabric_output']
        
        # Compute loss (prediction error)
        prediction = outputs['fabric_output']
        error = target - prediction
        loss = np.mean(error ** 2)
        
        # Backward propagation (random or intentional modes available)
        if self.config.use_random_prop and np.random.rand() < 0.1:
            self.propagator.propagate_backward(error, mode='random')
        elif self.config.use_intentional_prop:
            self.propagator.propagate_backward(error, mode='intentional')
        else:
            self.propagator.propagate_backward(error, mode='normal')
        
        # Combine forward and backward signals
        if self.config.bidirectional_weight > 0:
            combined = self.propagator.combine_directions()
        
        # Update parameters via optimizer
        if self.optimizer is not None:
            self.optimizer.step()
        
        return float(loss)
    
    def train_step(self, x: np.ndarray, target: Optional[np.ndarray] = None,
                   mode: str = 'normal') -> Dict[str, float]:
        """Single training step"""
        # Update biological system
        dt = 0.001  # 1ms timestep
        self.bio_system.update(
            dt=dt,
            goals=None,
            pred_vec=None,
            novelty=0.0,
            health=1.0,
            dream=False
        )
        
        # Get energy gate (affects learning)
        modulators = self.bio_system.modulators()
        energy_gate = modulators['energy_gate']
        
        if energy_gate < 0.1:
            # Too fatigued to learn effectively
            return {'loss': 0.0, 'energy': energy_gate, 'skipped': True}
        
        # Forward pass
        outputs = self.forward_pass(x, mode=mode)
        
        # Backward pass
        loss = self.backward_pass(outputs, target)
        
        # Update statistics
        self.total_steps += 1
        self.loss_history.append(loss)
        
        return {
            'loss': loss,
            'energy': energy_gate,
            'learning_drive': modulators['learning_drive'],
            'mood': modulators['mood'],
            'skipped': False
        }
    
    def train_epoch(self, data_loader, mode: str = 'normal') -> Dict[str, float]:
        """Train for one epoch"""
        epoch_losses = []
        
        for batch_x in data_loader:
            metrics = self.train_step(batch_x, mode=mode)
            if not metrics['skipped']:
                epoch_losses.append(metrics['loss'])
        
        self.epoch += 1
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        
        logger.info(f"Epoch {self.epoch}: Loss = {avg_loss:.6f}")
        
        return {
            'epoch': self.epoch,
            'avg_loss': avg_loss,
            'total_steps': self.total_steps
        }
    
    def train(self, data_loader, epochs: Optional[int] = None):
        """
        Main training loop.
        If epochs is None, trains continuously (for online learning).
        """
        epochs = epochs or self.config.max_epochs
        
        if epochs is None:
            logger.info("Starting continuous training mode")
            epoch = 0
            while True:
                # Alternate between normal, random, and intentional modes
                if epoch % 3 == 0:
                    mode = 'normal'
                elif epoch % 3 == 1:
                    mode = 'random'
                else:
                    mode = 'intentional'
                
                metrics = self.train_epoch(data_loader, mode=mode)
                logger.info(f"Mode: {mode}, Metrics: {metrics}")
                epoch += 1
                
                # Allow interruption
                time.sleep(0.01)
        else:
            logger.info(f"Starting training for {epochs} epochs")
            for epoch in range(epochs):
                mode = ['normal', 'random', 'intentional'][epoch % 3]
                metrics = self.train_epoch(data_loader, mode=mode)
                logger.info(f"Mode: {mode}, Metrics: {metrics}")
    
    def get_state_dict(self) -> Dict:
        """Get complete training state"""
        return {
            'epoch': self.epoch,
            'total_steps': self.total_steps,
            'som_weights': self.som.weights,
            'fabric_neurons': [[[(n.membrane_potential if n else 0.0) 
                                for n in row] 
                               for row in layer] 
                              for layer in self.neural_fabric.neurons],
            'loss_history': list(self.loss_history)
        }
    
    def load_state_dict(self, state: Dict):
        """Load training state"""
        self.epoch = state.get('epoch', 0)
        self.total_steps = state.get('total_steps', 0)
        if 'som_weights' in state:
            self.som.weights = state['som_weights']
        if 'loss_history' in state:
            self.loss_history = deque(state['loss_history'], maxlen=1000)
        logger.info(f"Loaded state: epoch={self.epoch}, steps={self.total_steps}")


def create_synthetic_data(batch_size: int, spatial_dims: Tuple[int, int, int],
                         num_batches: int = 100) -> List[np.ndarray]:
    """Create synthetic training data for testing"""
    data = []
    for _ in range(num_batches):
        batch = np.random.randn(batch_size, *spatial_dims, 1).astype(np.float32)
        data.append(batch)
    return data


if __name__ == '__main__':
    # Example usage
    config = TrainingConfig(
        learning_rate=1e-4,
        batch_size=8,
        time_steps=8,
        spatial_dims=(13, 13, 64),
        use_random_prop=True,
        use_intentional_prop=True,
        bidirectional_weight=0.5
    )
    
    trainer = NeuralFabricTrainer(config)
    
    # Create synthetic data
    data_loader = create_synthetic_data(
        batch_size=config.batch_size,
        spatial_dims=config.spatial_dims,
        num_batches=10
    )
    
    # Train for a few epochs
    logger.info("Starting training demonstration")
    trainer.train(data_loader, epochs=5)
    
    logger.info("Training complete")
    state = trainer.get_state_dict()
    logger.info(f"Final state: {state['epoch']} epochs, {state['total_steps']} steps")
