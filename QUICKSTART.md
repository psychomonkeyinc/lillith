# Quick Start Guide

This guide will help you get Lillith up and running quickly.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Webcam and microphone (optional, can run without)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/psychomonkeyinc/lillith.git
cd lillith
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- numpy (numerical computing)
- opencv-python (video processing)
- sounddevice (audio processing)
- psutil (system monitoring)
- PyQt5 (user interface)
- scipy (scientific computing)

### 3. Verify Installation

Run the test suite to ensure everything is working:

```bash
python run_tests.py
```

You should see: `✅ All tests passed!` (or close to it)

## Running Lillith

### Option 1: Basic Run (Default)

```bash
python main.py
```

This will:
- Start the UI
- Run for 60 seconds awake
- Run for 60 seconds in dream mode
- Automatically save and shutdown

### Option 2: Custom Duration

```bash
# Run for 2 minutes awake, 1 minute dream
LILLITH_AWAKE_SEC=120 LILLITH_DREAM_SEC=60 python main.py
```

### Option 3: Alternative Launcher

```bash
python run.py
```

This launches the UI first, then the cognitive system.

## Device Selection

On first run, you'll be prompted to select:
1. **Video device** (camera)
2. **Audio device** (microphone)

You can skip either by entering `-1`.

## Monitoring

The system provides:
- **Console logs**: Real-time text output
- **UI dashboard**: Visual representation of cognitive state
- **Data collection**: Saved to `data_collection/` directory

## Common Issues

### "No module named 'numpy'"

Install dependencies:
```bash
pip install -r requirements.txt
```

### "No video devices found"

- Check camera connection
- On Linux, you may need to run as root or add user to `video` group
- You can run without video by selecting `-1` when prompted

### "No audio devices found"

- Check microphone connection
- On Linux, install `portaudio`: `sudo apt-get install portaudio19-dev`
- You can run without audio by selecting `-1` when prompted

### UI doesn't start

Make sure PyQt5 is installed:
```bash
pip install PyQt5
```

On some Linux systems you may need:
```bash
sudo apt-get install python3-pyqt5
```

## Testing

### Run All Tests

```bash
python run_tests.py
```

### Run Individual Test Suites

```bash
python test_nn.py          # Neural network tests
python test_emotion.py     # Emotion system tests
python test_memory.py      # Memory system tests
python test_mind.py        # Cognitive processing tests
python test_som.py         # Self-organizing map tests
python test_integration.py # Integration tests
```

## Configuration

### Environment Variables

- `LILLITH_AWAKE_SEC`: Awake duration (default: 60)
- `LILLITH_DREAM_SEC`: Dream duration (default: 60)
- `LILLITH_SLOW_INIT`: Delay between init stages in seconds (default: 0)
- `PYTHONDONTWRITEBYTECODE`: Prevent .pyc files (set to "1")

### Example

```bash
# Run for 5 minutes awake, no dream phase
LILLITH_AWAKE_SEC=300 LILLITH_DREAM_SEC=0 python main.py
```

## Data Output

Lillith generates several output files:

- `main_log.txt`: Detailed system logs
- `trainedcafve.pkl`: Trained feature encoder state
- `data_collection/`: Raw sensory and cognitive data

## Stopping Lillith

### Graceful Shutdown

- Press `Ctrl+C` in the terminal
- Or create a file named `STOP_LILLITH` in the project directory

### Force Stop

- Press `Ctrl+C` twice (not recommended - may lose state)

## Next Steps

1. Read the full [README.md](README.md) for detailed information
2. Review [IMPROVEMENTS.md](IMPROVEMENTS.md) for enhancement ideas
3. Explore the module source code
4. Try modifying parameters to see how behavior changes

## Getting Help

If you encounter issues:

1. Check the console output for error messages
2. Review `main_log.txt` for detailed logs
3. Consult the [IMPROVEMENTS.md](IMPROVEMENTS.md) troubleshooting section
4. Open an issue on GitHub with:
   - Your Python version
   - Operating system
   - Complete error message
   - Steps to reproduce

## Architecture Overview

```
User Input → Sensory Processing → Feature Extraction → 
Consciousness Encoding → Cognitive Mapping → Emotional Processing →
Memory Integration → Decision Making → Output Generation
```

Key components:
- **SFE**: Sensory Feature Extractor (audio/video)
- **CAFVE**: Consciousness-Aware Feature Vector Encoder
- **SOM**: Self-Organizing Map (cognitive mapping)
- **Mind**: Unified cognitive state processor
- **Emotion**: Multi-dimensional emotional system
- **Memory**: Storage, retrieval, and consolidation
- **Output**: Language and action generation

## Performance Tips

1. **Reduce dimensions** if running on limited hardware
2. **Disable UI** by commenting out display initialization
3. **Shorter cycles** with reduced `AWAKE_SEC` and `DREAM_SEC`
4. **Skip video** if you don't need visual processing

## Development

To contribute or modify:

1. Create a new branch
2. Make your changes
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

Happy exploring!
