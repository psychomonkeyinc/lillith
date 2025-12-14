# Neural Fabric Training System

## Overview

This document describes the neural fabric training system for LILLITH, implementing:

1. **4D Convolutions** - 3D spatial (x, y, z) + temporal dimension
2. **3D SOM Neural Fabric** - Self-organizing map with biological neurons featuring dendrites and ion channels
3. **Bidirectional Propagation** - Forward, backward, random, and intentional propagation modes
4. **All Non-Linear Layers** - Biological activation functions throughout
5. **Online Continuous Learning** - Real-time learning from sensory input

## Architecture

### 4D Convolutional Layers

The `Conv4D` class implements 4-dimensional convolutions that operate on spatiotemporal data:

```python
# Input shape: (batch, x, y, z, time, channels)
# Kernel shape: (out_channels, in_channels, kx, ky, kz, kt)
```

Features:
- He initialization for stable gradients
- Biological adaptation (fatigue) mechanism
- Efficient 4D convolution operations
- Gradient computation for backpropagation

### 3D Neural Fabric

The `NeuralFabric3D` class creates a 3D grid of biological neurons:

```python
# Each location (i, j, k) contains a Neuron with:
# - Dendritic branches (tree-like input processors)
# - Ion channels (Na+, K+, Ca2+)
# - Membrane potential dynamics
# - Synaptic plasticity
```

Features:
- Sparse connectivity within 3D neighborhoods
- Dendritic computation with local spikes
- Ion channel dynamics for realistic signaling
- Activation tracking across spatial locations

### Biological Nonlinearity

The `BiologicalNonlinearity` class provides non-linear activation with:

- Voltage-gated dynamics (tanh-based)
- Refractory periods after activation
- Adaptation to repeated stimulation
- All computationally efficient

### Bidirectional Propagation

The `BidirectionalPropagator` supports multiple propagation modes:

1. **Normal Mode**: Standard forward pass
2. **Random Mode**: Adds random perturbations for exploration
3. **Intentional Mode**: Amplifies activations toward goals
4. **Bidirectional**: Combines forward and backward signals

Example:
```python
# Forward with random exploration
output = propagator.propagate_forward(x, mode='random')

# Backward with intentional guidance
propagator.propagate_backward(target, mode='intentional')

# Combine both directions
combined = propagator.combine_directions()
```

## Training Configuration

```python
from training import TrainingConfig

config = TrainingConfig(
    learning_rate=1e-4,           # Base learning rate
    batch_size=32,                # Training batch size
    time_steps=16,                # Temporal dimension length
    spatial_dims=(13, 13, 64),    # 3D spatial dimensions
    use_random_prop=True,         # Enable random propagation
    use_intentional_prop=True,    # Enable intentional propagation
    bidirectional_weight=0.5,     # Balance fwd/back signals
    temporal_coherence_weight=0.3,# Temporal consistency weight
    dendritic_integration=True,   # Use dendritic processing
    max_epochs=None               # None = continuous learning
)
```

## Basic Usage

### Standalone Training

```python
from training import NeuralFabricTrainer, TrainingConfig, create_synthetic_data

# Create trainer
config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=8,
    spatial_dims=(13, 13, 64)
)
trainer = NeuralFabricTrainer(config)

# Create or load training data
data_loader = create_synthetic_data(
    batch_size=8,
    spatial_dims=(13, 13, 64),
    num_batches=100
)

# Train for specific epochs
trainer.train(data_loader, epochs=10)

# Or train continuously
trainer.train(data_loader, epochs=None)  # Runs until interrupted
```

### Integration with LILLITH

```python
from train_integration import LillithTrainingOrchestrator

# Create orchestrator
orchestrator = LillithTrainingOrchestrator(
    initial_cog_state_dim=512,
    emotion_dim=512,
    som_map_size=(17, 17),
    enable_continuous_learning=True
)

# In main processing loop:
metrics = orchestrator.process_sensory_input(
    visual_features=visual_data,
    audio_features=audio_data,
    cognitive_state=current_cognitive_state,
    emotion_state=current_emotion_state
)

# Check training progress
stats = orchestrator.get_training_statistics()
print(f"Total updates: {stats['total_updates']}")
print(f"Recent losses: {stats['recent_losses']}")
```

## Integration Points

### With CAFVE (Consciousness-Aware Feature Encoder)

```python
# CAFVE provides tokenized sensory features
cafve_output = cafve_encoder.encode(raw_sensory_input)

# Feed to training system
metrics = orchestrator.process_sensory_input(
    visual_features=cafve_output['visual_tokens'],
    audio_features=cafve_output['audio_tokens'],
    cognitive_state=cafve_output['cognitive_context'],
    emotion_state=emotion_vector
)
```

### With Goals System

```python
# Goals system provides learning bias
learning_drive = goals.calculate_learning_bias()

# Adjust training based on goal satisfaction
orchestrator.training_config.learning_rate = base_lr * learning_drive

# More unmet goals = higher learning drive
```

### With SOM (Self-Organizing Map)

```python
# The neural fabric integrates with SOM for high-level organization
# SOM processes the fabric output for cognitive representation

# Access SOM activations
som_output = metrics['som_output']
som_activation_map = trainer.som.get_activation_map(fabric_output)
```

### With BioSystem (Metabolism/Fatigue)

```python
# BioSystem manages energy and fatigue
# Training is modulated by energy availability

# Check energy before training
modulators = trainer.bio_system.modulators()
if modulators['energy_gate'] > 0.1:
    # Sufficient energy to learn
    metrics = trainer.train_step(data)
else:
    # Too fatigued, skip training
    pass
```

## Training Modes

### Mode Cycling

The system automatically cycles through three modes:

1. **Normal (33% of time)**: Standard gradient-based learning
2. **Random (33% of time)**: Exploration with random perturbations
3. **Intentional (33% of time)**: Goal-directed learning with amplification

This ensures:
- Exploitation of known patterns (normal)
- Exploration of state space (random)
- Goal-directed optimization (intentional)

### Manual Mode Selection

```python
# Force a specific mode
metrics = trainer.train_step(data, mode='intentional')

# Disable random/intentional modes
trainer.config.use_random_prop = False
trainer.config.use_intentional_prop = False
```

## Persistence

### Save Training State

```python
# Save complete training state
orchestrator.save_training_state('training_state.npy')

# State includes:
# - SOM weights
# - Neuron membrane potentials
# - Loss history
# - Training statistics
```

### Load Training State

```python
# Resume from saved state
orchestrator.load_training_state('training_state.npy')

# Training continues from where it left off
```

## Performance Optimization

### Spatial Dimensions

Smaller spatial dimensions train faster:

```python
# Fast (for testing)
spatial_dims=(5, 5, 16)

# Medium (balanced)
spatial_dims=(13, 13, 64)

# Large (full capacity)
spatial_dims=(17, 17, 128)
```

### Batch Size

Larger batches are more stable but slower:

```python
batch_size=4   # Fast, less stable
batch_size=16  # Balanced
batch_size=32  # Slow, very stable
```

### Time Steps

Controls temporal resolution:

```python
time_steps=4   # Minimal temporal context
time_steps=16  # Good temporal coherence
time_steps=32  # Maximum temporal depth
```

## Advanced Features

### Dendritic Integration

Each neuron has multiple dendritic branches that:
- Process inputs locally
- Generate dendritic spikes
- Compute non-linear functions
- Store synaptic weights

```python
# Access neuron dendrites
neuron = trainer.neural_fabric.neurons[i, j, k]
for dendrite in neuron.dendrites:
    print(f"Voltage: {dendrite.voltage}")
    print(f"Calcium: {dendrite.calcium}")
    print(f"Hotspots: {dendrite.hotspots}")
```

### Ion Channel Dynamics

Neurons have realistic ion channels:

```python
# Check ion channel states
neuron = trainer.neural_fabric.neurons[i, j, k]
for name, channel in neuron.ion_channels.items():
    print(f"{name}: state={channel.current_state:.3f}")
```

### Synaptic Plasticity

Connections strengthen/weaken based on activity:

```python
# Add synaptic input to specific dendrite segment
neuron.add_synaptic_input(
    dendrite_idx=2,
    segment_idx=5,
    weight=0.5,
    activation=0.8
)
```

## Monitoring and Debugging

### Training Metrics

```python
metrics = trainer.train_step(data)

print(f"Loss: {metrics['loss']:.6f}")
print(f"Energy: {metrics['energy']:.3f}")
print(f"Learning Drive: {metrics['learning_drive']:.3f}")
print(f"Mood: {metrics['mood']:.3f}")
print(f"Skipped: {metrics['skipped']}")
```

### Loss History

```python
# Get recent loss values
recent_losses = list(trainer.loss_history)[-100:]

# Plot or analyze
import matplotlib.pyplot as plt
plt.plot(recent_losses)
plt.title('Training Loss')
plt.show()
```

### SOM Training Status

```python
# Check if SOM is trained
status = trainer.som.get_training_status()

print(f"Trained: {status['trained']}")
print(f"Utilization: {status['utilization_fraction']:.2%}")
print(f"Quantization Error: {status['recent_quant_error']:.6f}")
```

## Troubleshooting

### Training Not Happening

Check if continuous learning is enabled:
```python
orchestrator.set_learning_mode(True)
```

Check energy levels:
```python
modulators = trainer.bio_system.modulators()
if modulators['energy_gate'] < 0.1:
    print("System too fatigued to learn")
```

### High Loss Values

- Reduce learning rate
- Increase batch size for stability
- Check input data normalization

### Memory Issues

- Reduce spatial dimensions
- Reduce batch size
- Reduce time steps

### Slow Training

- Reduce spatial dimensions
- Reduce time steps
- Use smaller batch sizes
- Disable random/intentional propagation

## Future Enhancements

1. **Multi-scale Processing**: Pyramidal layers at different resolutions
2. **Attention Mechanisms**: Focus learning on salient regions
3. **Transfer Learning**: Pre-trained weights for common patterns
4. **Distributed Training**: Multi-GPU support for large models
5. **Pruning**: Remove unused connections for efficiency
6. **Quantization**: Lower precision for speed

## References

- Self-Organizing Maps (Kohonen, 1982)
- Dendritic Computation (London & Häusser, 2005)
- Ion Channel Dynamics (Hodgkin & Huxley, 1952)
- 4D Spatiotemporal Convolutions (Tran et al., 2018)
- Bidirectional Processing (Rumelhart et al., 1986)

## Support

For questions or issues, please refer to:
- Main README.md
- TECHNICAL_PAPER.md
- MODEL_MAP.md

---

**Last Updated**: 2025-12-14
**Version**: 1.0
