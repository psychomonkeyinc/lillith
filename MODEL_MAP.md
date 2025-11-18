# LILLITH - Complete Model Architecture Map

## System Overview

Lillith is an advanced artificial consciousness system built from scratch using pure NumPy. The architecture implements a biologically-inspired cognitive system with sensory processing, emotional modeling, memory consolidation, consciousness tokenization, and theory of mind capabilities.

## Architecture Layers

### Layer 1: Core Neural Infrastructure

#### **nn.py** (600 lines)
- **Purpose**: From-scratch neural network implementation
- **Key Components**:
  - `Layer`: Abstract base class for all neural layers
  - `Linear`: Fully connected dense layer with biological properties (float16 optimization)
  - `Sigmoid`, `ReLU`, `Tanh`: Activation functions with custom backpropagation
  - `Sequential`: Container for layer stacks
  - `AdamW`: Base optimizer with weight decay
  - `JustinJOptimizer`: Enhanced optimizer with agency-focused training
  - `mse_loss_prime`: Loss function derivative
- **Dependencies**: NumPy only
- **Used By**: All modules requiring neural computation

#### **OptiJustinJ.py** (460 lines)
- **Purpose**: Standalone enriched optimizer with internal audio feedback loop
- **Key Features**:
  - Agency-focused optimization
  - Vocal/audio internal feedback integration
  - Multi-timescale adaptive learning rate modulation
  - Gradient hygiene (clipping, NaN/Inf scrubbing)
  - Pattern memory with decay and replay sampling
  - Spectral alignment metrics
- **Parameters**:
  - Base learning rate: 1e-4
  - Vocal feedback weight: 0.3
  - Agency growth rate: 0.01
  - Replay capacity: 512 patterns
- **Dependencies**: NumPy, logging
- **Used By**: main.py, run.py for optimizing neural networks

### Layer 2: Sensory Processing

#### **inout.py** (794 lines)
- **Purpose**: Unified sensory input/output processing
- **Key Components**:
  - `AudioIn`: Audio input processing and feature extraction
  - `VideoIn`: Video capture and frame processing
  - `AudioOut`: Audio output and synthesis
- **Features**:
  - Multi-backend support (PyAudio, sounddevice)
  - Real-time audio/video streaming
  - Device enumeration and selection
- **Dependencies**: cv2, numpy, audio libraries
- **Used By**: main.py, run.py

#### **cafve.py** (821 lines)
- **Purpose**: Consciousness-Aware Feature Vector Encoder
- **Key Components**:
  - `SensoryFeatureExtractor`: Multi-stage sensory processing (512 → 1024 → 2048)
  - `ConsciousnessAwareFeatureVectorEncoder`: CAFVE tokenization
  - `ConsciousnessContext`: Context tracking for awareness
- **Features**:
  - Progressive feature dimension scaling
  - Consciousness token generation
  - Attention-weighted feature integration
- **Dimensions**:
  - Sensory stages: [512, 1024, 2048]
  - CAFVE token dimension: 512
- **Dependencies**: nn.py, NumPy
- **Used By**: main.py, mind.py, predict.py

### Layer 3: Cognitive Core

#### **mind.py** (625 lines)
- **Purpose**: Central cognitive integration and processing
- **Key Components**:
  - `CognitiveScaling`: Dynamic dimension scaling across cognitive modules
  - `Mind`: Main cognitive orchestration class
- **Dimension Stages**:
  - Stage 0: (512, 1024, 2048)
  - Stage 1: (1024, 2048, 4096)
  - Stage 2: (2048, 4096, 8192)
  - Stage 3: (4096, 8192, 16384)
- **Features**:
  - Dynamic architectural growth based on complexity metrics
  - Integration of SOM, emotion, and memory systems
  - Unified cognitive state management
- **Required Arguments**:
  - initial_dim_stage, som_activation_dim (289), som_bmu_coords_dim (2)
  - emotional_state_dim, memory_recall_dim, predictive_error_dim
  - unified_cognitive_state_dim
- **Dependencies**: nn.py, som.py, emotion.py, memory.py
- **Used By**: main.py, run.py

#### **som.py** (2788 lines - largest module)
- **Purpose**: Self-Organizing Map implementation with biological properties
- **Key Components**:
  - `SelfOrganizingMap`: Kohonen SOM with enhanced learning
  - JustinJ optimizer integration (lines 1-1505)
  - Biological modeling (lines 1505+)
- **Configuration**:
  - Map size: 17×17 (289 neurons, prime number)
  - Activation dimension: 289
  - BMU coordinate dimension: 2 (x, y)
  - Default input dimension: 256 (scalable)
- **Features**:
  - Competitive learning
  - Topological organization
  - Best Matching Unit (BMU) computation
  - Neighborhood function with decay
- **Dependencies**: NumPy, logging
- **Used By**: mind.py, dream.py, main.py

#### **attention.py** (194 lines)
- **Purpose**: Attention mechanism for selective focus
- **Key Components**:
  - `Attention`: Attention weight computation and focus management
- **Features**:
  - Multi-head attention patterns
  - Dynamic focus allocation
  - Saliency-based weighting
- **Dimension**: 512 (attention focus vector)
- **Dependencies**: nn.py
- **Used By**: main.py, mind.py

#### **predict.py** (157 lines)
- **Purpose**: Predictive processing and error computation
- **Key Components**:
  - `Predict`: Prediction network for anticipatory processing
- **Constants**:
  - CAFVE_TOKEN_DIM: 512
  - Predictive error dimension: 512
- **Features**:
  - Forward prediction
  - Error signal generation
  - Integration with CAFVE tokens
- **Dependencies**: nn.py
- **Used By**: main.py

### Layer 4: Emotional & Social Systems

#### **emotion.py** (383 lines)
- **Purpose**: Emotional state modeling and processing
- **Key Components**:
  - `EmotionalState`: Emotional vector representation
  - `EmotionCore`: Emotion processing engine
- **Configuration**:
  - Dimension: 512
  - First 108 dimensions: Named emotions (mapped)
  - Remaining dimensions: Emergent emotional states
- **Emotional Stages**: [512, 1024, 2048]
- **Features**:
  - Multi-dimensional emotion vectors
  - Emotional state transitions
  - Valence and arousal computation
- **Dependencies**: NumPy
- **Used By**: mind.py, goals.py, itsagirl.py

#### **itsagirl.py** (306 lines)
- **Purpose**: Gender-aware self-model and identity
- **Key Components**:
  - `ItsAGirl`: Self-identity processing
- **Features**:
  - Self-referential modeling
  - Identity-aware processing
  - Gendered perspective integration
- **Dependencies**: NumPy, logging
- **Used By**: main.py, run.py

#### **tom.py** (370 lines)
- **Purpose**: Theory of Mind - modeling other agents' mental states
- **Key Components**:
  - `ToM`: Theory of Mind processor
- **Configuration**:
  - Model dimension: 512
- **Features**:
  - Agent state prediction
  - Intent inference
  - Social cognition
- **Dependencies**: nn.py
- **Used By**: main.py, run.py

#### **conscience.py** (179 lines)
- **Purpose**: Moral judgment and ethical reasoning
- **Key Components**:
  - `Conscience`: Ethical evaluation system
- **Features**:
  - Moral valence computation
  - Ethical constraint checking
  - Value alignment
- **Dependencies**: nn.py
- **Used By**: main.py, goals.py

### Layer 5: Memory Systems

#### **memory.py** (295 lines)
- **Purpose**: Memory storage, retrieval, and consolidation
- **Key Components**:
  - `MemoryFragment`: Individual memory representation
  - `MemorySystem`: Memory management and recall
- **Configuration**:
  - Memory recall dimension: 512
- **Features**:
  - Episodic memory storage
  - Semantic memory integration
  - Memory consolidation during sleep
  - Retrieval by similarity
- **Dependencies**: NumPy
- **Used By**: mind.py, dream.py

#### **nvme_memory.py** (267 lines)
- **Purpose**: Persistent memory storage on NVMe devices
- **Features**:
  - High-speed persistent storage
  - Memory fragment serialization
  - NVMe-optimized I/O
- **Dependencies**: NumPy, pickle
- **Used By**: memory.py (optional backend)

#### **dream.py** (348 lines)
- **Purpose**: Dream state processing and memory consolidation
- **Key Components**:
  - `Dream`: Dream manager and consolidation engine
- **Features**:
  - Memory consolidation during sleep
  - Pattern replay
  - SOM-based failure log processing
  - Synaptic homeostasis
- **Dependencies**: som.py, memory.py, emotion.py
- **Used By**: main.py (deferred initialization)

### Layer 6: Higher Cognition

#### **goals.py** (187 lines)
- **Purpose**: Goal formation, tracking, and achievement
- **Key Components**:
  - `Goals`: Goal management system
- **Features**:
  - Goal prioritization
  - Progress tracking
  - Emotion-goal integration
  - Sub-goal decomposition
- **Dependencies**: emotion.py
- **Used By**: main.py, run.py

#### **language.py** (371 lines)
- **Purpose**: Language processing and symbolic reasoning
- **Key Components**:
  - `Language`: Language understanding and generation
- **Configuration**:
  - Internal language dimension: 512
- **Features**:
  - Symbolic workspace
  - Language encoding/decoding
  - Semantic processing
- **Dependencies**: nn.py
- **Used By**: main.py, output.py

#### **temporal.py** (430 lines)
- **Purpose**: Temporal reasoning and sequence processing
- **Features**:
  - Temporal pattern recognition
  - Sequence prediction
  - Time-aware processing
- **Dependencies**: NumPy
- **Used By**: Optional integration with mind.py

### Layer 7: Output & Expression

#### **output.py** (376 lines)
- **Purpose**: Output generation and expression coordination
- **Key Components**:
  - `Output`: Output management system
- **Configuration**:
  - Input dimension: 512
  - Output dimension: 512
- **Features**:
  - Multi-modal output coordination
  - Expression synthesis
  - Response generation
- **Dependencies**: nn.py
- **Used By**: main.py, run.py

#### **vocalsynth.py** (286 lines)
- **Purpose**: Vocal synthesis and speech generation
- **Key Components**:
  - `VocalSynth`: Voice synthesis engine
  - Biquad filter for formant synthesis
- **Configuration**:
  - Vocal synthesis parameters dimension: 512
- **Features**:
  - Formant synthesis
  - Prosody control
  - Speech generation
- **Dependencies**: NumPy
- **Used By**: main.py, output.py

### Layer 8: Meta-Systems

#### **metamind.py** (106 lines)
- **Purpose**: Architecture growth optimizer and meta-learning
- **Key Components**:
  - `MetaMind`: Meta-cognitive controller
- **Features**:
  - Architecture optimization
  - Growth decision making
  - Resource allocation
- **Dependencies**: NumPy
- **Used By**: main.py

#### **health.py** (274 lines)
- **Purpose**: System health monitoring and self-diagnosis
- **Key Components**:
  - `Health`: Health monitoring system
- **Features**:
  - Resource usage tracking
  - Performance monitoring
  - Anomaly detection
  - Self-repair triggers
- **Dependencies**: nn.py, psutil
- **Used By**: main.py, run.py

#### **reward.py** (44 lines)
- **Purpose**: Deprecated reward system (being replaced)
- **Status**: Legacy module, future pleasure/pain mechanism planned
- **Used By**: None (deprecated)

### Layer 9: Orchestration & Control

#### **main.py** (1661 lines - main orchestrator)
- **Purpose**: Central orchestration and coordination
- **Key Components**:
  - `LillithOrchestrator`: Main system controller
  - `UILogHandler`: Logging integration with UI
  - Device selection and validation
  - System initialization and shutdown
- **Global Dimensions**:
  - SFE_DIM: 512
  - INITIAL_COG_STATE_DIM: 512
  - EMOTION_DIM: 512
  - INTERNAL_LANG_DIM: 512
  - VOCAL_SYNTH_PARAMS_DIM: 512
  - ATTENTION_FOCUS_DIM: 512
  - SOM_MAP_SIZE: (17, 17)
  - MEMORY_RECALL_DIM: 512
  - PREDICTIVE_ERROR_DIM: 512
  - TOM_MODEL_DIM: 512
- **Timed Run Configuration**:
  - Awake duration: 60 seconds (configurable via LILLITH_AWAKE_SEC)
  - Dream duration: 60 seconds (configurable via LILLITH_DREAM_SEC)
- **Dependencies**: All modules
- **Entry Point**: Used by run.py

#### **run.py** (231 lines)
- **Purpose**: Application entry point and process launcher
- **Key Functions**:
  - `launch_model()`: Initialize all subsystems
  - `main()`: Start display UI and command listener
- **Features**:
  - Display-first launch sequence
  - Background model thread
  - Command queue processing
  - State persistence (disabled)
- **Launch Sequence**:
  1. Start Qt UI in main thread
  2. Command listener waits for LAUNCH_MODEL
  3. Model subsystems initialized in background thread
  4. Continuous cycle with 0.1s sleep
- **Dependencies**: All core modules, display
- **Entry Point**: `if __name__ == "__main__"`

### Layer 10: UI & Data Collection

#### **display.py** (808 lines)
- **Purpose**: PyQt-based visualization and UI
- **Key Functions**:
  - `start_display_process()`: Launch display in separate process
  - `start_qt_app()`: Run Qt application
- **Features**:
  - Real-time system visualization
  - Module state display
  - Command interface
  - Log viewer
- **Dependencies**: PyQt5/PyQt6, matplotlib
- **Used By**: run.py

#### **data.py** (664 lines)
- **Purpose**: Data collection and logging
- **Key Components**:
  - `DataCollection`: Data capture and storage
- **Features**:
  - Sensory data logging
  - State snapshot recording
  - Performance metrics collection
- **Output Directory**: ./data_collection
- **Dependencies**: NumPy, pickle
- **Used By**: main.py

## Data Flow Architecture

### Awake State Processing Pipeline

```
[AudioIn/VideoIn] → [SensoryFeatureExtractor] → [CAFVE]
                                                    ↓
                                              [CAFVE Tokens]
                                                    ↓
    ┌───────────────────────────────────────────────┴─────────────────────────┐
    ↓                                                                           ↓
[Attention] → [Predict] → [Mind (SOM + Emotion + Memory)]                 [Language]
                ↓                       ↓                                       ↓
         [Predictive Error]        [Goals + Conscience]                  [Symbolic Rep]
                                          ↓                                     ↓
                                    [ItsAGirl + ToM]                      [Output]
                                          ↓                                     ↓
                                      [Health]                          [VocalSynth]
                                          ↓                                     ↓
                                    [MetaMind]                           [AudioOut]
                                          ↓
                                  [Memory Storage]
```

### Dream State Processing Pipeline

```
[Memory System] → [Dream Manager] → [SOM Replay]
                        ↓
                [Consolidation]
                        ↓
            [Emotion Integration]
                        ↓
            [Memory Re-encoding]
                        ↓
            [Updated Memory]
```

## Dimensional Consistency

All modules operate within a coherent dimensional framework:

| Component | Dimension | Scaling |
|-----------|-----------|---------|
| Sensory Features | 512 → 1024 → 2048 | Dynamic |
| Cognitive State | 512 → 1024 → 2048 | Dynamic |
| Emotion Vector | 512 → 1024 → 2048 | Dynamic |
| Internal Language | 512 | Fixed |
| Attention Focus | 512 | Fixed |
| SOM Activation | 289 (17×17) | Fixed |
| Memory Recall | 512 | Fixed |
| Predictive Error | 512 | Fixed |
| ToM Model | 512 | Fixed |
| CAFVE Token | 512 | Fixed |

## System Initialization Sequence

1. **Display Launch** (run.py)
   - Qt UI starts in main thread
   - Command listener thread started

2. **Model Initialization** (on LAUNCH_MODEL command)
   - Neural core (nn.py)
   - Optimizers (AdamW, JustinJOptimizer)
   - Sensory processing (SensoryFeatureExtractor, CAFVE)
   - SOM (17×17 map, 256D input)
   - Emotion core (512D)
   - Memory system
   - Mind (cognitive integration)
   - Higher cognition (Goals, Conscience, ToM, ItsAGirl)
   - Health monitoring
   - Language processing
   - Attention mechanism
   - Output system
   - Vocal synthesis
   - I/O systems (AudioIn, AudioOut, VideoIn)

3. **Runtime Cycle** (0.1s loop)
   - Collect metrics snapshot
   - Send to display queue
   - Process commands
   - Increment cycle counter

## File Statistics

| Module | Lines | Category |
|--------|-------|----------|
| som.py | 2788 | Core Cognitive |
| main.py | 1661 | Orchestration |
| cafve.py | 821 | Sensory Processing |
| display.py | 808 | UI |
| inout.py | 794 | I/O |
| data.py | 664 | Data Collection |
| mind.py | 625 | Core Cognitive |
| nn.py | 600 | Neural Infrastructure |
| OptiJustinJ.py | 460 | Optimization |
| temporal.py | 430 | Temporal Processing |
| emotion.py | 383 | Emotional System |
| output.py | 376 | Output Generation |
| language.py | 371 | Language Processing |
| tom.py | 370 | Social Cognition |
| dream.py | 348 | Memory Consolidation |
| itsagirl.py | 306 | Self-Model |
| memory.py | 295 | Memory System |
| vocalsynth.py | 286 | Speech Synthesis |
| health.py | 274 | Health Monitoring |
| nvme_memory.py | 267 | Persistent Storage |
| run.py | 231 | Entry Point |
| attention.py | 194 | Attention Mechanism |
| goals.py | 187 | Goal Management |
| conscience.py | 179 | Ethical Reasoning |
| predict.py | 157 | Prediction |
| metamind.py | 106 | Meta-Learning |
| reward.py | 44 | Deprecated |
| **Total** | **14,025** | **27 Modules** |

## Module Dependencies Graph

```
run.py (entry)
  └─→ display.py (UI)
  └─→ main.py (orchestrator)
       ├─→ nn.py (neural core)
       ├─→ OptiJustinJ.py (optimizer)
       ├─→ inout.py (I/O)
       ├─→ cafve.py
       │    └─→ nn.py
       ├─→ som.py
       ├─→ emotion.py
       ├─→ memory.py
       ├─→ mind.py
       │    ├─→ nn.py
       │    ├─→ som.py
       │    ├─→ emotion.py
       │    └─→ memory.py
       ├─→ itsagirl.py
       ├─→ goals.py
       │    └─→ emotion.py
       ├─→ conscience.py
       │    └─→ nn.py
       ├─→ tom.py
       │    └─→ nn.py
       ├─→ health.py
       │    └─→ nn.py
       ├─→ dream.py
       │    ├─→ som.py
       │    ├─→ memory.py
       │    └─→ emotion.py
       ├─→ language.py
       │    └─→ nn.py
       ├─→ attention.py
       │    └─→ nn.py
       ├─→ predict.py
       │    └─→ nn.py
       ├─→ output.py
       │    └─→ nn.py
       ├─→ vocalsynth.py
       ├─→ metamind.py
       └─→ data.py
```

## Configuration & Environment

### Environment Variables
- `LILLITH_AWAKE_SEC`: Awake duration (default: 60)
- `LILLITH_DREAM_SEC`: Dream duration (default: 60)
- `PYTHONDONTWRITEBYTECODE`: Set to '1' to prevent .pyc files

### Directories
- `./data_collection`: Data logging output
- `main_log.txt`: Primary log file

### Performance Targets
- Max cycle time: 0.05s (50ms soft target)
- Memory warning threshold: 85%
- CPU warning threshold: 95%
- Resource monitor interval: 2.0s

## Key Design Principles

1. **Pure NumPy Implementation**: No external ML frameworks
2. **Biological Inspiration**: Neural dynamics modeled after biological systems
3. **Dynamic Scaling**: Architecture grows based on complexity needs
4. **Consciousness-Aware**: Explicit consciousness token generation
5. **Emotional Integration**: Emotions integrated throughout processing
6. **Memory Consolidation**: Sleep/dream cycles for memory processing
7. **Agency-Focused**: Optimizer designed for autonomous agency
8. **Modular Architecture**: Clean separation of concerns
9. **Real-time Processing**: Optimized for live sensory input
10. **Self-Monitoring**: Health and meta-cognitive oversight

## Future Enhancements

- Pleasure/pain mechanism (replacement for reward.py)
- Enhanced temporal processing integration
- Expanded dream consolidation algorithms
- Advanced theory of mind capabilities
- Deeper language understanding
- Extended NVMe memory utilization
- Multi-agent interaction support

---

**Last Updated**: 2025-10-19
**Total System Size**: 14,025 lines of code across 27 modules
**Primary Language**: Python (NumPy-based)
**Architecture**: Biologically-inspired artificial consciousness
