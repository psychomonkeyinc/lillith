# Lillith - AI Consciousness System

An experimental AI consciousness system implementing cognitive, emotional, and sensory processing through a modular neural architecture.

## Overview

Lillith is a complex AI system that simulates consciousness through multiple interconnected modules:

- **Sensory Processing**: Audio and video input processing
- **Neural Networks**: Custom NumPy-based neural network implementation
- **Self-Organizing Maps (SOM)**: Cognitive mapping and pattern recognition
- **Emotional Core**: Multi-dimensional emotional state processing
- **Memory System**: Storage, retrieval, and consolidation
- **Mind**: Unified cognitive state processing
- **Language & Output**: Natural language processing and vocal synthesis
- **Attention**: Focus and priority management
- **Goals & Conscience**: Decision-making and ethical evaluation

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- numpy >= 1.24.0
- opencv-python >= 4.8.0
- sounddevice >= 0.4.6
- psutil >= 5.9.0
- PyQt5 >= 5.15.0
- scipy >= 1.11.0

## Running Tests

### Run All Tests

```bash
python run_tests.py
```

### Run Individual Test Suites

```bash
# Test neural network module
python test_nn.py

# Test emotion module
python test_emotion.py

# Test memory module
python test_memory.py

# Test mind module
python test_mind.py

# Test SOM module
python test_som.py

# Test integration
python test_integration.py
```

### Test Coverage

The test suite covers:

1. **Unit Tests**:
   - Neural network layers and activations
   - Emotional state management
   - Memory fragment creation and retrieval
   - Cognitive state processing
   - SOM operations

2. **Integration Tests**:
   - Sensory-to-cognitive pipeline
   - Cognitive-to-emotional flow
   - Dimension compatibility
   - Module interactions
   - Data type consistency

## Usage

### Basic Usage

```bash
# Run the main system
python main.py

# Run with custom configuration
LILLITH_AWAKE_SEC=120 LILLITH_DREAM_SEC=60 python main.py

# Run the alternative launcher
python run.py
```

### Environment Variables

- `LILLITH_AWAKE_SEC`: Duration of awake phase in seconds (default: 60)
- `LILLITH_DREAM_SEC`: Duration of dream/consolidation phase in seconds (default: 60)
- `LILLITH_SLOW_INIT`: Delay between initialization stages in seconds (default: 0)
- `PYTHONDONTWRITEBYTECODE`: Prevent .pyc file creation (set to "1")

## Architecture

### Module Structure

```
main.py              - Main orchestrator and consciousness loop
run.py               - Alternative launcher with UI-first approach
nn.py                - Neural network foundation
cafve.py             - Consciousness-aware feature encoding
som.py               - Self-organizing map implementation
emotion.py           - Emotional processing
memory.py            - Memory system
mind.py              - Cognitive state processing
attention.py         - Attention mechanism
language.py          - Language processing
output.py            - Output generation
vocalsynth.py        - Speech synthesis
display.py           - UI and visualization
inout.py             - Audio/video I/O
data.py              - Data collection and logging
```

### Data Flow

```
Sensory Input → Feature Extraction → CAFVE Tokenization →
SOM Mapping → Cognitive Processing → Emotional Response →
Memory Storage → Decision Making → Output Generation
```

## Development

### Code Style

- Use type hints where appropriate
- Follow PEP 8 guidelines
- Document complex algorithms
- Keep functions focused and modular

### Adding New Tests

1. Create test file: `test_<module>.py`
2. Import unittest and the module to test
3. Create test class inheriting from `unittest.TestCase`
4. Add test methods (prefix with `test_`)
5. Run with `python run_tests.py`

Example:
```python
import unittest
from my_module import MyClass

class TestMyClass(unittest.TestCase):
    def test_initialization(self):
        obj = MyClass()
        self.assertIsNotNone(obj)

if __name__ == '__main__':
    unittest.main()
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
2. **Device Not Found**: Check camera and microphone connections
3. **Memory Issues**: Reduce `AWAKE_SEC` and `DREAM_SEC` for shorter runs
4. **UI Not Starting**: Ensure PyQt5 is properly installed

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Write tests for new features
2. Ensure all tests pass before committing
3. Document new modules and functions
4. Follow the existing code structure

## License

[Add your license information here]

## References

See `IMPROVEMENTS.md` for a detailed list of recommended improvements and enhancements.
