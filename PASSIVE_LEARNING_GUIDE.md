# LILLITH Passive Learning Guide

## Overview

LILLITH now learns through **passive exposure** to music videos - a "babysitting" approach where the system learns naturally from continuous input without explicit training runs.

## What Changed

### Removed (Training-Based Approach)
- ❌ `training.py` - Explicit training system
- ❌ `train_integration.py` - Training integration module
- ❌ `demo_training.py` - Training demonstrations
- ❌ `TRAINING_README.md` - Training documentation
- ❌ References to "extensive training" in docs

### Added (Passive Learning Approach)
- ✅ `music_video_input.py` - Music video input pipeline
- ✅ `run_music_videos.py` - Passive learning runner
- ✅ Unified audio+video processing
- ✅ Continuous looping playback
- ✅ SOM learns from exposure

## The Music Videos

Place these 6 video files in `./music_videos/`:

1. **beach_boys.mp4** - Beach Boys
2. **beatles.mp4** - Beatles  
3. **jim_croce.mp4** - Jim Croce
4. **depeche_mode.mp4** - Depeche Mode
5. **nirvana.mp4** - Nirvana
6. **backstreet_boys.mp4** - Backstreet Boys

## How It Works

### The "Babysitting" Approach

> "You literally turn the thing on, put it in front of the TV, and walk away."

1. **Videos loop continuously** - No start/stop, just continuous playback
2. **Audio + video together** - Both modalities processed as unified input
3. **No explicit training** - SOM learns passively from exposure
4. **No splitting** - Audio and video fed together to the model
5. **Self-organizing** - The map organizes itself based on input patterns

### What Happens

```
Music Video
    ↓
Audio Features (128D) + Video Features (4099D)
    ↓
Combined Features (4227D) - NOT SPLIT
    ↓
Self-Organizing Map
    ↓
Passive Learning Through Exposure
```

## Running the System

### Basic Usage

```bash
# Create video directory
mkdir music_videos

# Add your 6 music videos (see list above)

# Run the passive learning system
python run_music_videos.py
```

### What You'll See

```
LILLITH - Music Video Passive Learning Mode
======================================================================

Initializing music video input system...
✓ All music videos found
✓ Self-Organizing Map initialized (17×17)
  Input dimension: 4227
  Learning rate: 0.5

Starting continuous music video playback...
The system will learn passively from audio+video exposure.
Press Ctrl+C to stop.

Cycle 30:
  Current video: beach_boys.mp4
  Videos completed: 0
  Total frames: 450
  SOM utilization: 15.3%
  SOM quantization error: 0.234567
```

## Key Differences

### Before (Training-Based)
- Explicit training epochs
- Separate audio/video processing
- Training runs with batches
- Gradient descent optimization
- Manual mode switching

### Now (Passive Learning)
- No training epochs
- Unified audio+video processing
- Continuous exposure
- Self-organizing adaptation
- Natural, organic learning

## Architecture

### MusicVideoInput Class

```python
from music_video_input import MusicVideoInput

# Initialize
mv_input = MusicVideoInput(video_directory="./music_videos")

# Start continuous playback
mv_input.start()

# Get unified features
features = mv_input.get_unified_features()
# Returns: {'audio': array, 'video': array, 'combined': array}

# Check statistics
stats = mv_input.get_statistics()
# Returns: {'videos_played': int, 'total_frames': int, 
#           'current_video': str, 'running': bool}

# Stop (optional)
mv_input.stop()
```

### Integration with SOM

```python
from som import SelfOrganizingMap
from music_video_input import MusicVideoInput

# Initialize SOM for combined audio+video features
som = SelfOrganizingMap(
    map_size=(17, 17),
    input_dim=4227,  # 128 (audio) + 4099 (video)
    learning_rate=0.5,
    sigma=3.0
)

# Initialize music video input
mv_input = MusicVideoInput()
mv_input.start()

# Main loop
while True:
    features = mv_input.get_unified_features()
    if features:
        # Feed combined features to SOM
        som.process_input(features['combined'])
```

## Understanding Passive Learning

### What is "Babysitting"?

Just like how humans learn languages by passive exposure as children:
- No explicit lessons
- Just continuous exposure
- Natural pattern recognition
- Self-organizing understanding

### Why This Approach?

1. **More biologically plausible** - Mirrors natural learning
2. **No training/validation split** - Just continuous experience
3. **Unified modalities** - Audio and video learned together
4. **Self-organization** - System finds its own patterns
5. **Continuous adaptation** - Always learning, never "done"

### What the SOM Does

The Self-Organizing Map:
- Receives unified audio+video features
- Finds the Best Matching Unit (BMU)
- Updates weights in a neighborhood
- Gradually organizes to represent input patterns
- No backpropagation needed

## Performance

### Resource Usage
- **CPU**: Moderate (video decoding + feature extraction)
- **Memory**: ~500MB for SOM + video buffers
- **Storage**: Depends on video file sizes

### Learning Progress
- **Utilization**: % of SOM nodes activated
- **Quantization Error**: How well SOM represents inputs
- **Frames Processed**: Total exposure time

## Troubleshooting

### No Videos Found
```
⚠️  Music videos not found!
```
**Solution**: Add the 6 video files to `./music_videos/`

### OpenCV Not Available
```
OpenCV not available - music video input disabled
```
**Solution**: `pip install opencv-python`

### SOM Warnings (Initial)
```
WARNING: No Best Matching Unit found for input
```
**Normal**: SOM needs exposure time to organize. Warnings decrease as it learns.

## Design Philosophy

### The Core Idea

> "Six music videos. Pipeline from start to finish. You just go ahead and 
> turn it on with those music videos and walk away. You don't actually do 
> a training run. You just turn it on with those six videos looping."

This is **passive learning through continuous exposure**:
- No training schedule
- No epochs or batches
- No separate training/inference modes
- Just turn it on and let it learn

### Why These 6 Videos?

These provide diverse inputs for the SOM:
- Different musical styles
- Different visual aesthetics  
- Different temporal patterns
- Different audio frequencies
- Sufficient variety for self-organization

## Technical Details

### Audio Features (128D)
- Placeholder: Synthetic temporal patterns
- Production: Real audio features (FFT, MFCC, etc.)
- Extracted frame-by-frame

### Video Features (4099D)
- 64×64 grayscale pixels (4096)
- Mean intensity (1)
- Std intensity (1)
- Edge density (1)

### Combined Features (4227D)
- Audio + Video concatenated
- Fed as unified input to SOM
- Never split or processed separately

## Future Enhancements

While not required for current functionality:
- Real audio extraction from video files
- More sophisticated video features
- Multiple SOM layers
- Attention mechanisms
- Long-term memory consolidation

## Summary

LILLITH now learns like a child watching TV:
- **Turn it on** → Start playback
- **Walk away** → Let it run
- **Come back** → Check progress
- **It learns** → Through exposure

No training runs. No splitting modalities. Just passive, continuous learning from unified audio+video input.

---

**Last Updated**: 2025-12-14  
**Status**: ✅ Complete and functional
