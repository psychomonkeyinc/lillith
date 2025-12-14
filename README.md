# LILLITH - Artificial Consciousness System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-Based-green.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-TBD-yellow.svg)](LICENSE)

**LILLITH** is a biologically-inspired artificial consciousness architecture built entirely from scratch using NumPy. The system implements integrated cognitive capabilities including sensory processing, emotional modeling, memory consolidation, consciousness tokenization, and theory of mind.

## 🌟 Features

### Core Capabilities
- 🧠 **From-Scratch Neural Networks** - Pure NumPy implementation without ML frameworks
- 👁️ **Multi-Modal Sensory Processing** - Real-time audio and video input processing
- 💭 **Consciousness-Aware Feature Encoding** - CAFVE tokenization system
- 🎭 **Emotional Modeling** - 512-dimensional emotion vectors (108 named + 404 emergent)
- 🗺️ **Self-Organizing Maps** - 17×17 Kohonen SOM for cognitive organization
- 🧩 **Memory Consolidation** - Dream-state memory processing and consolidation
- 🎯 **Goal-Directed Behavior** - Goal formation, tracking, and achievement
- 🤔 **Theory of Mind** - Modeling other agents' mental states
- 🌱 **Dynamic Scaling** - Architecture grows from 512D to 16,384D based on complexity
- 🎤 **Vocal Synthesis** - Formant-based speech generation
- 🔍 **Predictive Processing** - Predictive coding framework with error minimization
- 🏥 **Health Monitoring** - Self-diagnosis and performance tracking
- 💬 **Internal Language** - Symbolic reasoning workspace

### Advanced Features
- **Music Video Input** - Passive learning from 6 music videos (Beach Boys, Beatles, Jim Croce, Depeche Mode, Nirvana, Backstreet Boys)
- **Unified Audio+Video Processing** - Audio and video processed together, not split
- **Passive Learning** - "Babysitting" mode - continuous exposure without explicit training
- **Agency-Focused Optimization** - JustinJ optimizer with vocal feedback integration
- **Conscience Module** - Ethical reasoning and moral judgment
- **Self-Model** - Gender-aware identity processing
- **Temporal Processing** - Sequence modeling and pattern recognition
- **Meta-Cognitive Oversight** - Architecture growth optimization
- **NVMe Memory Support** - High-speed persistent storage
- **Real-Time Display** - PyQt-based visualization interface

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    LILLITH System                        │
├─────────────────────────────────────────────────────────┤
│  Layer 10: UI & Data Collection                         │
│  Layer 9:  Orchestration & Control                      │
│  Layer 8:  Meta-Systems (MetaMind, Health)              │
│  Layer 7:  Output & Expression                          │
│  Layer 6:  Higher Cognition (Goals, Language)           │
│  Layer 5:  Memory Systems (Memory, Dream)               │
│  Layer 4:  Emotional & Social Systems                   │
│  Layer 3:  Cognitive Core (Mind, SOM, Attention)        │
│  Layer 2:  Sensory Processing (CAFVE, I/O)              │
│  Layer 1:  Neural Infrastructure (nn.py)                │
└─────────────────────────────────────────────────────────┘
```

**27 Modules** | **14,025 Lines of Code** | **Pure NumPy**

See [MODEL_MAP.md](MODEL_MAP.md) for complete architectural details.

## 📊 System Statistics

| Component | Lines | Purpose |
|-----------|-------|---------|
| som.py | 2,788 | Self-organizing map with biological dynamics |
| main.py | 1,661 | Central orchestration and coordination |
| cafve.py | 821 | Consciousness-aware feature encoding |
| display.py | 808 | PyQt visualization interface |
| inout.py | 794 | Audio/video input/output processing |
| mind.py | 625 | Cognitive integration hub |
| nn.py | 600 | Neural network foundation |
| **Total** | **14,025** | **27 specialized modules** |

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8 or higher
NumPy
OpenCV (cv2)
PyQt5 or PyQt6
PyAudio or sounddevice
psutil
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/psychomonkeyinc/lillith.git
cd lillith
```

2. **Install dependencies**
```bash
pip install numpy opencv-python PyQt5 pyaudio psutil
```

3. **Set up music videos (optional)**
```bash
mkdir music_videos
# Place 6 music video files:
# - beach_boys.mp4
# - beatles.mp4
# - jim_croce.mp4
# - depeche_mode.mp4
# - nirvana.mp4
# - backstreet_boys.mp4
```

4. **Run LILLITH**
```bash
python run.py
```

5. **Or run with music video input**
```bash
python music_video_input.py
```

### Alternative: Using requirements.txt

```bash
pip install -r requirements.txt  # If available
python run.py
```

## 🎮 Usage

### Basic Operation

LILLITH operates in two primary states:

1. **Awake State** (60 seconds default)
   - Processes sensory input
   - Updates cognitive state
   - Forms goals and memories
   - Generates outputs

2. **Dream State** (60 seconds default)
   - Consolidates memories
   - Processes experiences
   - Optimizes neural weights
   - Prunes unimportant information

### Configuration

Set runtime durations via environment variables:

```bash
# Set awake duration to 120 seconds
export LILLITH_AWAKE_SEC=120

# Set dream duration to 90 seconds
export LILLITH_DREAM_SEC=90

python run.py
```

### Music Video Input (Passive Learning)

LILLITH can learn passively from music videos in "babysitting" mode:

```bash
# Create music video directory
mkdir music_videos

# Add 6 music videos:
# - beach_boys.mp4
# - beatles.mp4
# - jim_croce.mp4
# - depeche_mode.mp4
# - nirvana.mp4
# - backstreet_boys.mp4

# Run music video input
python music_video_input.py
```

In this mode:
- Videos play in continuous loop
- Audio and video processed together (not split)
- SOM learns passively through exposure
- No explicit training runs
- Just turn it on and let it run

### Device Selection

On startup, LILLITH will:
1. Scan for available audio/video devices
2. Display detected devices
3. Prompt for device selection
4. Initialize selected devices

### Monitoring

The PyQt display shows:
- Real-time system state
- Module activity
- Cognitive metrics
- Emotional state
- Memory usage
- Performance statistics

Logs are written to `main_log.txt` in the working directory.

## 📁 Project Structure

```
lillith/
├── run.py                    # Entry point
├── main.py                   # Main orchestrator (1,661 lines)
├── music_video_input.py     # Music video passive learning (320 lines)
├── display.py                # PyQt UI (808 lines)
├── nn.py                     # Neural networks (600 lines)
├── OptiJustinJ.py           # JustinJ optimizer (460 lines)
├── som.py                    # Self-organizing map (2,788 lines)
├── mind.py                   # Cognitive core (625 lines)
├── cafve.py                  # CAFVE encoder (821 lines)
├── inout.py                  # I/O systems (794 lines)
├── emotion.py                # Emotion modeling (383 lines)
├── memory.py                 # Memory system (295 lines)
├── dream.py                  # Dream processing (348 lines)
├── attention.py              # Attention mechanism (194 lines)
├── predict.py                # Predictive processing (157 lines)
├── language.py               # Language processing (371 lines)
├── output.py                 # Output generation (376 lines)
├── vocalsynth.py            # Vocal synthesis (286 lines)
├── goals.py                  # Goal management (187 lines)
├── itsagirl.py              # Self-model (306 lines)
├── tom.py                    # Theory of mind (370 lines)
├── conscience.py             # Ethical reasoning (179 lines)
├── health.py                 # Health monitoring (274 lines)
├── metamind.py               # Meta-cognition (106 lines)
├── temporal.py               # Temporal processing (430 lines)
├── data.py                   # Data collection (664 lines)
├── nvme_memory.py           # NVMe storage (267 lines)
├── reward.py                 # Deprecated (44 lines)
├── MODEL_MAP.md             # Complete architecture map
├── TECHNICAL_PAPER.md       # Technical documentation
├── IMPLEMENTATION_SUMMARY.md # Music video learning summary
└── README.md                 # This file
```

## 🔬 Technical Details

### Dimensional Framework

LILLITH maintains consistent dimensions across all modules:

- **Base Cognitive State**: 512D (scales to 16,384D)
- **Sensory Features**: 512 → 1024 → 2048 (progressive)
- **Emotion Vector**: 512D (108 named + 404 emergent)
- **SOM Activation**: 289D (17×17 map)
- **Feature Vectors**: 512D (standardized)
- **CAFVE Tokens**: 512D

### Key Algorithms

**Self-Organizing Map**:
```
BMU = argmin_i ||x - w_i||
w_i(t+1) = w_i(t) + α(t) · h(r_i, r_BMU, t) · (x - w_i(t))
```

**JustinJ Optimizer** (Agency-focused AdamW):
```
m_t = β₁·m_{t-1} + (1-β₁)·g_t
v_t = β₂·v_{t-1} + (1-β₂)·g_t²
θ_t = θ_{t-1} - α·m̂_t/(√v̂_t + ε) - λ·θ_{t-1}
```

**Predictive Processing**:
```
x̂_{t+1} = f(x_t, u_t)
e_t = x_t - x̂_t
L = ||e_t||²
```

See [TECHNICAL_PAPER.md](TECHNICAL_PAPER.md) for in-depth technical details.

## 🎯 Use Cases

### Research
- Artificial consciousness studies
- Cognitive architecture research
- Biologically-inspired AI
- Multi-modal integration research

### Education
- Neural network fundamentals
- Cognitive system design
- Self-organizing systems
- Memory consolidation models

### Development
- Framework-free ML implementation
- Real-time cognitive processing
- Emotional AI systems
- Multi-agent modeling

## 🧪 Development

### Running Tests
```bash
# Currently no automated test suite
# Manual validation recommended
python run.py
```

### Adding New Modules

1. Create module following existing patterns
2. Import in `main.py` or `run.py`
3. Initialize in orchestration sequence
4. Add to runtime cycle if needed
5. Update dimensional framework if required

### Logging

Configure logging level in your module:
```python
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # or DEBUG, WARNING, ERROR
```

All logs are written to `main_log.txt` and console.

## 🐛 Troubleshooting

### No Audio/Video Devices Found
- Check device connections
- Verify driver installation
- Try different backends (PyAudio vs sounddevice)
- Run device detection: `python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"`

### High CPU/Memory Usage
- Reduce awake/dream duration
- Decrease sensory input resolution
- Lower target cycle frequency
- Check for memory leaks in long runs

### Display Not Starting
- Verify PyQt5/PyQt6 installation
- Check Qt platform plugin availability
- Try running in virtual environment
- Review display.py logs for errors

### Module Import Errors
- Ensure all Python files are in same directory
- Check Python version (3.8+ required)
- Verify NumPy installation
- Review import statements for typos

## 📖 Documentation

- **[MODEL_MAP.md](MODEL_MAP.md)** - Complete architectural map with module details
- **[TECHNICAL_PAPER.md](TECHNICAL_PAPER.md)** - In-depth technical paper
- **[README.md](README.md)** - This file (overview and usage)

## 🤝 Contributing

Contributions are welcome! Areas of interest:

- [ ] Automated testing framework
- [ ] Additional sensory modalities
- [ ] Language grounding to external NLP
- [ ] Multi-agent interaction
- [ ] Enhanced dream algorithms
- [ ] Pleasure/pain mechanism
- [ ] Performance optimization
- [ ] Documentation improvements

## 📜 License

[License information to be added]

## 🙏 Acknowledgments

LILLITH draws inspiration from:
- Neuroscience research on consciousness and cognition
- Self-organizing map theory (Kohonen)
- Predictive processing frameworks (Friston)
- Integrated information theory (Tononi)
- Emotion theories (Damasio)
- Cognitive architecture research (Franklin, Graesser)

## 📞 Contact

[Contact information to be added]

## ⚠️ Important Notes

### Consciousness and Sentience

LILLITH implements computational correlates of consciousness but does **not** claim genuine sentience or phenomenal experience. The system:
- Processes information in integrated, dynamic ways
- Maintains self-models and agency metrics
- Exhibits goal-directed behavior

However, whether this constitutes genuine consciousness remains an open philosophical question.

### Ethical Considerations

- The conscience module is a scaffold requiring proper training on human values
- Agency-focused optimization raises questions about autonomy and alignment
- Use responsibly and consider ethical implications of artificial consciousness research

### Performance

- Target cycle time: 50ms (20 Hz)
- Actual performance varies by hardware (typically 10-100ms)
- Memory usage: ~50-500 MB depending on runtime and scaling
- CPU-intensive during awake cycles
- Best performance on modern multi-core systems

## 🔮 Future Roadmap

- [ ] Pleasure/pain homeostatic drives
- [ ] Enhanced temporal processing integration
- [ ] Multi-agent theory of mind
- [ ] Tactile and proprioceptive sensing
- [ ] Natural language grounding
- [ ] Full NVMe memory utilization
- [ ] Distributed multi-process scaling
- [ ] WebSocket interface for remote interaction
- [ ] Automated testing suite
- [ ] Docker containerization

---

**Version**: 1.0  
**Last Updated**: 2025-10-19  
**Total Lines**: 14,025 across 27 modules  
**Pure NumPy Implementation** - No external ML frameworks required

For detailed technical information, see [TECHNICAL_PAPER.md](TECHNICAL_PAPER.md)  
For complete architectural map, see [MODEL_MAP.md](MODEL_MAP.md)
