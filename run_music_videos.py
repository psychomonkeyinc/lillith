#!/usr/bin/env python3
"""
run_music_videos.py

Run LILLITH with music video input for passive learning.
This demonstrates the "babysitting" approach where the system
learns from continuous exposure to music videos without explicit training.
"""

import sys
import time
import logging
import numpy as np
from music_video_input import MusicVideoInput, create_example_playlist_file

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_with_music_videos():
    """
    Run LILLITH with music video input.
    
    This is a simplified version that focuses on the music video input
    and passive learning through the SOM.
    """
    
    logger.info("=" * 70)
    logger.info("LILLITH - Music Video Passive Learning Mode")
    logger.info("=" * 70)
    
    # Create music video directory and README if needed
    create_example_playlist_file()
    
    # Initialize music video input
    logger.info("\nInitializing music video input system...")
    mv_input = MusicVideoInput(video_directory="./music_videos")
    
    # Check if videos exist
    import os
    videos_exist = all(
        os.path.exists(os.path.join("./music_videos", video))
        for video in mv_input.playlist
    )
    
    if not videos_exist:
        logger.warning("\n⚠️  Music videos not found!")
        logger.info("\nPlease add the following 6 music video files to ./music_videos/:")
        for i, video in enumerate(mv_input.playlist, 1):
            logger.info(f"  {i}. {video}")
        logger.info("\nSee ./music_videos/README.txt for details.")
        logger.info("\nRunning in demo mode with synthetic features...")
        run_demo = True
    else:
        logger.info("✓ All music videos found")
        run_demo = False
    
    # Initialize SOM (if available)
    try:
        from som import SelfOrganizingMap
        som = SelfOrganizingMap(
            map_size=(17, 17),
            input_dim=4227,  # Combined audio+video features
            learning_rate=0.5,
            sigma=3.0
        )
        logger.info("✓ Self-Organizing Map initialized (17×17)")
        logger.info(f"  Input dimension: {som.input_dim}")
        logger.info(f"  Learning rate: {som.initial_lr}")
    except Exception as e:
        logger.warning(f"Could not initialize SOM: {e}")
        som = None
    
    # Start music video playback
    logger.info("\nStarting continuous music video playback...")
    logger.info("The system will learn passively from audio+video exposure.")
    logger.info("Press Ctrl+C to stop.\n")
    
    if not run_demo:
        mv_input.start()
    
    try:
        cycle = 0
        while True:
            cycle += 1
            
            # Get unified audio+video features
            if run_demo:
                # Demo mode: generate synthetic features
                audio_feat = np.random.randn(128).astype(np.float32)
                video_feat = np.random.randn(4099).astype(np.float32)
                features = {
                    'audio': audio_feat,
                    'video': video_feat,
                    'combined': np.concatenate([audio_feat, video_feat])
                }
            else:
                features = mv_input.get_unified_features()
            
            if features and som:
                # Feed to SOM for passive learning
                som.process_input(features['combined'])
                
                # Log progress periodically
                if cycle % 30 == 0:
                    stats = mv_input.get_statistics() if not run_demo else {
                        'current_video': 'demo',
                        'videos_played': cycle // 30,
                        'total_frames': cycle,
                        'running': True
                    }
                    
                    # Get SOM status
                    som_status = som.get_training_status()
                    
                    logger.info(f"Cycle {cycle}:")
                    logger.info(f"  Current video: {stats['current_video']}")
                    logger.info(f"  Videos completed: {stats['videos_played']}")
                    logger.info(f"  Total frames: {stats['total_frames']}")
                    logger.info(f"  SOM utilization: {som_status['utilization_fraction']:.1%}")
                    logger.info(f"  SOM quantization error: {som_status['recent_quant_error']:.6f}")
            
            # Small delay to control cycle rate
            time.sleep(0.033)  # ~30 fps
            
    except KeyboardInterrupt:
        logger.info("\n\nStopping music video playback...")
    finally:
        if not run_demo:
            mv_input.stop()
        
        logger.info("\n" + "=" * 70)
        logger.info("Session Summary")
        logger.info("=" * 70)
        
        if not run_demo:
            final_stats = mv_input.get_statistics()
            logger.info(f"Videos played: {final_stats['videos_played']}")
            logger.info(f"Total frames processed: {final_stats['total_frames']}")
        else:
            logger.info(f"Demo cycles: {cycle}")
        
        if som:
            som_status = som.get_training_status()
            logger.info(f"\nSOM Learning:")
            logger.info(f"  BMU hits: {som_status['total_bmu_hits']:.0f}")
            logger.info(f"  Utilization: {som_status['utilization_fraction']:.1%}")
            logger.info(f"  Quantization error: {som_status['recent_quant_error']:.6f}")
        
        logger.info("\n✓ Passive learning session complete")


if __name__ == '__main__':
    try:
        run_with_music_videos()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
