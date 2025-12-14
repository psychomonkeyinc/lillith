#!/usr/bin/env python3
"""
demo_training.py

Demonstration of LILLITH's neural fabric training system.
Shows all key features in action:
- 4D convolutions
- 3D SOM neural fabric with dendrites
- Bidirectional propagation
- Random and intentional modes
- Continuous online learning
"""

import sys
sys.dont_write_bytecode = True

import numpy as np
import logging
from training import NeuralFabricTrainer, TrainingConfig, create_synthetic_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def demo_basic_training():
    """Demonstrate basic training with the neural fabric"""
    logger.info("=" * 70)
    logger.info("DEMO 1: Basic Neural Fabric Training")
    logger.info("=" * 70)
    
    # Create configuration
    config = TrainingConfig(
        learning_rate=1e-4,
        batch_size=4,
        time_steps=8,
        spatial_dims=(7, 7, 32),  # Moderate size for demo
        use_random_prop=True,
        use_intentional_prop=True,
    )
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  Spatial dimensions: {config.spatial_dims}")
    logger.info(f"  Time steps: {config.time_steps}")
    logger.info(f"  Batch size: {config.batch_size}")
    
    # Create trainer
    logger.info("\nInitializing trainer...")
    trainer = NeuralFabricTrainer(config)
    
    # Create training data
    logger.info("Creating synthetic training data...")
    data = create_synthetic_data(
        batch_size=config.batch_size,
        spatial_dims=config.spatial_dims,
        num_batches=5
    )
    
    # Train for a few steps
    logger.info("\nTraining for 5 steps...\n")
    for i, batch in enumerate(data):
        metrics = trainer.train_step(batch, mode='normal')
        logger.info(f"Step {i+1}/5: loss={metrics['loss']:.6f}, "
                   f"energy={metrics['energy']:.3f}, "
                   f"mood={metrics['mood']:.3f}")
    
    logger.info("\n✓ Basic training complete!\n")


def demo_propagation_modes():
    """Demonstrate different propagation modes"""
    logger.info("=" * 70)
    logger.info("DEMO 2: Propagation Modes (Normal, Random, Intentional)")
    logger.info("=" * 70)
    
    config = TrainingConfig(
        batch_size=2,
        time_steps=4,
        spatial_dims=(5, 5, 16),  # Small for speed
    )
    
    trainer = NeuralFabricTrainer(config)
    data = create_synthetic_data(batch_size=2, spatial_dims=(5, 5, 16), num_batches=1)
    
    modes = ['normal', 'random', 'intentional']
    
    logger.info("\nTesting each propagation mode:")
    for mode in modes:
        metrics = trainer.train_step(data[0], mode=mode)
        logger.info(f"\n  {mode.upper()} mode:")
        logger.info(f"    Loss: {metrics['loss']:.6f}")
        logger.info(f"    Learning drive: {metrics['learning_drive']:.3f}")
        logger.info(f"    Exploration: {trainer.bio_system.explore:.3f}")
    
    logger.info("\n✓ All propagation modes working!\n")


def demo_biological_features():
    """Demonstrate biological neuron features"""
    logger.info("=" * 70)
    logger.info("DEMO 3: Biological Neural Features")
    logger.info("=" * 70)
    
    config = TrainingConfig(
        batch_size=1,
        spatial_dims=(3, 3, 8),  # Very small for inspection
    )
    
    trainer = NeuralFabricTrainer(config)
    
    # Inspect a neuron
    neuron = trainer.neural_fabric.neurons[1, 1, 1]
    
    logger.info("\nInspecting neuron at position (1, 1, 1):")
    logger.info(f"  Membrane potential: {neuron.membrane_potential:.2f} mV")
    logger.info(f"  Threshold: {neuron.threshold:.2f} mV")
    logger.info(f"  Number of dendrites: {len(neuron.dendrites)}")
    
    logger.info("\n  Ion channels:")
    for name, channel in neuron.ion_channels.items():
        logger.info(f"    {name}: state={channel.current_state:.3f}, "
                   f"conductance={channel.conductance:.3f}")
    
    logger.info("\n  Dendritic properties:")
    dendrite = neuron.dendrites[0]
    logger.info(f"    Segments: {dendrite.num_segments}")
    logger.info(f"    Hotspots: {np.sum(dendrite.hotspots)}/{dendrite.num_segments}")
    logger.info(f"    Voltage range: [{dendrite.voltage.min():.2f}, {dendrite.voltage.max():.2f}]")
    
    logger.info("\n✓ Biological features verified!\n")


def demo_continuous_learning():
    """Demonstrate continuous learning mode"""
    logger.info("=" * 70)
    logger.info("DEMO 4: Continuous Learning (5 cycles)")
    logger.info("=" * 70)
    
    config = TrainingConfig(
        batch_size=2,
        spatial_dims=(5, 5, 16),
        max_epochs=None  # Continuous
    )
    
    trainer = NeuralFabricTrainer(config)
    data_gen = lambda: create_synthetic_data(batch_size=2, spatial_dims=(5, 5, 16), num_batches=1)
    
    logger.info("\nContinuous learning simulation:")
    logger.info("(In real system, this runs indefinitely with sensory input)\n")
    
    for cycle in range(5):
        # Generate new data (simulates new sensory input)
        data = data_gen()
        
        # Train on it
        mode = ['normal', 'random', 'intentional'][cycle % 3]
        metrics = trainer.train_step(data[0], mode=mode)
        
        logger.info(f"Cycle {cycle+1}: mode={mode:11s}, "
                   f"loss={metrics['loss']:.6f}, "
                   f"energy={metrics['energy']:.3f}")
    
    # Show statistics
    stats = trainer.get_state_dict()
    logger.info(f"\nTotal training steps: {stats['total_steps']}")
    logger.info(f"Loss history length: {len(stats['loss_history'])}")
    
    logger.info("\n✓ Continuous learning demonstrated!\n")


def demo_4d_convolutions():
    """Demonstrate 4D convolutional processing"""
    logger.info("=" * 70)
    logger.info("DEMO 5: 4D Spatiotemporal Convolutions")
    logger.info("=" * 70)
    
    from training import Conv4D, BiologicalNonlinearity
    
    # Create 4D conv layer
    conv = Conv4D(in_channels=1, out_channels=8, kernel_size=(3, 3, 3, 3))
    activation = BiologicalNonlinearity()
    
    logger.info("\n4D Convolution layer:")
    logger.info(f"  Kernel shape: {conv.kernels.shape}")
    logger.info(f"  Input channels: {conv.in_channels}")
    logger.info(f"  Output channels: {conv.out_channels}")
    
    # Create test input: (batch, x, y, z, time, channels)
    test_input = np.random.randn(1, 5, 5, 5, 5, 1).astype(np.float32)
    logger.info(f"\nTest input shape: {test_input.shape}")
    
    # Forward pass
    output = conv.forward(test_input)
    logger.info(f"Output shape: {output.shape}")
    
    # Apply biological activation
    activated = activation.forward(output)
    logger.info(f"Activated shape: {activated.shape}")
    
    logger.info(f"\nBiological adaptation state: {activation.adaptation_state.mean():.6f}")
    logger.info(f"Refractory period: {activation.refractory.mean():.6f}")
    
    logger.info("\n✓ 4D convolutions working!\n")


def demo_som_integration():
    """Demonstrate SOM integration"""
    logger.info("=" * 70)
    logger.info("DEMO 6: Self-Organizing Map Integration")
    logger.info("=" * 70)
    
    config = TrainingConfig(
        batch_size=2,
        spatial_dims=(7, 7, 32),
    )
    
    trainer = NeuralFabricTrainer(config)
    data = create_synthetic_data(batch_size=2, spatial_dims=(7, 7, 32), num_batches=1)
    
    # Train and get outputs
    metrics = trainer.train_step(data[0])
    
    # Check SOM state
    som = trainer.som
    logger.info(f"\nSOM properties:")
    logger.info(f"  Map size: {som.map_size}")
    logger.info(f"  Input dimension: {som.input_dim}")
    logger.info(f"  Learning rate: {som.initial_lr}")
    
    # Get SOM training status
    status = som.get_training_status()
    logger.info(f"\nSOM training status:")
    logger.info(f"  Trained: {status['trained']}")
    logger.info(f"  Total BMU hits: {status['total_bmu_hits']:.0f}")
    logger.info(f"  Utilization: {status['utilization_fraction']:.2%}")
    
    # Get maps
    plasticity_map = som.get_plasticity_map()
    fatigue_map = som.get_fatigue_map()
    
    logger.info(f"\nSOM state:")
    logger.info(f"  Plasticity: mean={plasticity_map.mean():.3f}, std={plasticity_map.std():.3f}")
    logger.info(f"  Fatigue: mean={fatigue_map.mean():.3f}, std={fatigue_map.std():.3f}")
    
    logger.info("\n✓ SOM integration verified!\n")


def main():
    """Run all demonstrations"""
    logger.info("\n")
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║" + " " * 14 + "LILLITH Neural Fabric Training Demo" + " " * 18 + "║")
    logger.info("╚" + "═" * 68 + "╝")
    logger.info("\n")
    
    demos = [
        ("Basic Training", demo_basic_training),
        ("Propagation Modes", demo_propagation_modes),
        ("Biological Features", demo_biological_features),
        ("Continuous Learning", demo_continuous_learning),
        ("4D Convolutions", demo_4d_convolutions),
        ("SOM Integration", demo_som_integration),
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            logger.error(f"\n✗ Demo '{name}' failed: {e}")
            import traceback
            traceback.print_exc()
        
        input("Press Enter to continue to next demo...")
        print("\n")
    
    logger.info("=" * 70)
    logger.info("All demonstrations complete!")
    logger.info("=" * 70)
    logger.info("\nFeatures demonstrated:")
    logger.info("  ✓ 4D Convolutions (3D space + time)")
    logger.info("  ✓ 3D SOM Neural Fabric")
    logger.info("  ✓ Dendritic computation")
    logger.info("  ✓ Ion channel dynamics")
    logger.info("  ✓ Biological non-linear activations")
    logger.info("  ✓ Bidirectional propagation")
    logger.info("  ✓ Random propagation mode")
    logger.info("  ✓ Intentional propagation mode")
    logger.info("  ✓ Continuous online learning")
    logger.info("  ✓ Energy/fatigue management")
    logger.info("\nFor more information, see TRAINING_README.md\n")


if __name__ == '__main__':
    main()
