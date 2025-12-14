#!/usr/bin/env python3
"""
music_video_input.py

Music video input pipeline for LILLITH's continuous learning.
Processes 6 music videos in a loop, feeding audio and video together
as unified inputs into the self-organizing map for passive learning.

Music Videos:
1. Beach Boys
2. Beatles
3. Jim Croce
4. Depeche Mode
5. Nirvana
6. Backstreet Boys
"""

import os
import sys
import numpy as np
import logging
import threading
import time
from typing import Optional, List, Tuple, Dict
from collections import deque

try:
    import cv2
except ImportError:
    cv2 = None
    logging.warning("OpenCV not available - music video input disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MusicVideoInput:
    """
    Handles continuous looping playback of music videos for passive learning.
    Audio and video are processed together as unified inputs to the SOM.
    """
    
    def __init__(self, video_directory: str = "./music_videos"):
        """
        Initialize music video input pipeline.
        
        Args:
            video_directory: Directory containing the 6 music video files
        """
        self.video_directory = video_directory
        self.running = False
        self.current_video_index = 0
        
        # Music video playlist (in order)
        self.playlist = [
            "beach_boys.mp4",
            "beatles.mp4",
            "jim_croce.mp4",
            "depeche_mode.mp4",
            "nirvana.mp4",
            "backstreet_boys.mp4"
        ]
        
        # Current video capture
        self.cap = None
        self.current_video_path = None
        
        # Audio extraction (from video file)
        self.audio_sample_rate = 44100
        self.audio_chunk_size = 1024
        
        # Feature buffers for unified processing
        self.audio_features = deque(maxlen=100)
        self.video_features = deque(maxlen=100)
        
        # Threading
        self.thread = None
        self.lock = threading.Lock()
        
        # Statistics
        self.videos_played = 0
        self.total_frames_processed = 0
        
        logger.info(f"MusicVideoInput initialized with {len(self.playlist)} videos")
        logger.info(f"Video directory: {video_directory}")
    
    def start(self):
        """Start the music video playback loop"""
        if self.running:
            logger.warning("MusicVideoInput already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.thread.start()
        logger.info("MusicVideoInput started - continuous playback enabled")
    
    def stop(self):
        """Stop the music video playback"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        logger.info("MusicVideoInput stopped")
    
    def _playback_loop(self):
        """Main playback loop - continuously plays videos in sequence"""
        while self.running:
            # Get next video in playlist
            video_name = self.playlist[self.current_video_index]
            video_path = os.path.join(self.video_directory, video_name)
            
            # Check if video file exists
            if not os.path.exists(video_path):
                logger.warning(f"Video not found: {video_path}")
                logger.info(f"Skipping to next video...")
                self._next_video()
                time.sleep(0.1)
                continue
            
            # Load and play video
            logger.info(f"Now playing: {video_name}")
            self._play_video(video_path)
            
            # Move to next video
            self._next_video()
            self.videos_played += 1
    
    def _play_video(self, video_path: str):
        """Play a single video file, processing audio and video together"""
        if cv2 is None:
            logger.error("OpenCV not available - cannot play video")
            return
        
        # Open video file
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return
        
        self.current_video_path = video_path
        frame_count = 0
        
        # Get video properties
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default fallback
        
        frame_delay = 1.0 / fps
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                # Video finished
                logger.info(f"Finished playing video (processed {frame_count} frames)")
                break
            
            # Process frame
            self._process_frame(frame, frame_count)
            frame_count += 1
            self.total_frames_processed += 1
            
            # Maintain approximate frame rate
            time.sleep(frame_delay)
        
        # Release video
        self.cap.release()
        self.cap = None
    
    def _process_frame(self, frame: np.ndarray, frame_number: int):
        """
        Process a single video frame along with corresponding audio.
        Audio and video are kept together as unified input.
        """
        with self.lock:
            # Extract visual features from frame
            visual_features = self._extract_visual_features(frame)
            
            # Extract audio features (from video audio track)
            # Note: For simplicity, we extract audio-like features from video metadata
            # In a full implementation, would use proper audio extraction
            audio_features = self._extract_audio_features(frame_number)
            
            # Store unified features
            self.video_features.append(visual_features)
            self.audio_features.append(audio_features)
    
    def _extract_visual_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract visual features from video frame.
        Returns a feature vector suitable for SOM input.
        """
        # Resize frame to standard size
        resized = cv2.resize(frame, (64, 64))
        
        # Convert to grayscale
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
        
        # Compute simple statistics
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Compute edge features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Flatten and normalize
        flat = gray.flatten().astype(np.float32) / 255.0
        
        # Combine features
        features = np.concatenate([
            flat,
            [mean_intensity / 255.0, std_intensity / 255.0, edge_density]
        ])
        
        return features
    
    def _extract_audio_features(self, frame_number: int) -> np.ndarray:
        """
        Extract audio features corresponding to current video frame.
        In a full implementation, this would extract from the video's audio track.
        """
        # Placeholder: Generate synthetic audio features based on frame number
        # In real implementation, would use proper audio extraction (e.g., ffmpeg)
        
        # Simple spectral-like features
        num_features = 128
        features = np.zeros(num_features, dtype=np.float32)
        
        # Simulate some temporal variation
        phase = frame_number * 0.01
        for i in range(num_features):
            features[i] = 0.5 * (1.0 + np.sin(phase + i * 0.1))
        
        return features
    
    def _next_video(self):
        """Move to next video in playlist (wraps around)"""
        self.current_video_index = (self.current_video_index + 1) % len(self.playlist)
    
    def get_unified_features(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get current unified audio+video features for SOM input.
        Returns both audio and video as a unified representation.
        """
        with self.lock:
            if not self.audio_features or not self.video_features:
                return None
            
            # Get most recent features
            audio = self.audio_features[-1]
            video = self.video_features[-1]
            
            # Return as unified dictionary
            return {
                'audio': audio,
                'video': video,
                'combined': np.concatenate([audio, video])
            }
    
    def get_statistics(self) -> Dict:
        """Get playback statistics"""
        return {
            'videos_played': self.videos_played,
            'total_frames': self.total_frames_processed,
            'current_video': self.playlist[self.current_video_index],
            'running': self.running
        }


def create_example_playlist_file(video_directory: str = "./music_videos"):
    """
    Create a README file with instructions for adding music videos.
    """
    os.makedirs(video_directory, exist_ok=True)
    
    readme_path = os.path.join(video_directory, "README.txt")
    with open(readme_path, 'w') as f:
        f.write("Music Video Playlist for LILLITH\n")
        f.write("=" * 50 + "\n\n")
        f.write("Place the following 6 music video files in this directory:\n\n")
        f.write("1. beach_boys.mp4 - Beach Boys music video\n")
        f.write("2. beatles.mp4 - Beatles music video\n")
        f.write("3. jim_croce.mp4 - Jim Croce music video\n")
        f.write("4. depeche_mode.mp4 - Depeche Mode music video\n")
        f.write("5. nirvana.mp4 - Nirvana music video\n")
        f.write("6. backstreet_boys.mp4 - Backstreet Boys music video\n\n")
        f.write("These videos will play in a continuous loop, providing\n")
        f.write("unified audio+video input for passive learning.\n\n")
        f.write("The system learns by continuous exposure ('babysitting'),\n")
        f.write("not through explicit training runs.\n")
    
    logger.info(f"Created playlist README at: {readme_path}")


# Example usage
if __name__ == '__main__':
    logger.info("Music Video Input System Test")
    
    # Create example directory and README
    create_example_playlist_file()
    
    # Initialize system
    mv_input = MusicVideoInput(video_directory="./music_videos")
    
    # Start playback
    logger.info("Starting music video playback...")
    mv_input.start()
    
    # Run for a short test period
    try:
        for i in range(10):
            time.sleep(1)
            features = mv_input.get_unified_features()
            if features:
                logger.info(f"Got unified features: audio={features['audio'].shape}, "
                          f"video={features['video'].shape}, "
                          f"combined={features['combined'].shape}")
            
            stats = mv_input.get_statistics()
            logger.info(f"Stats: {stats}")
    except KeyboardInterrupt:
        logger.info("Test interrupted")
    finally:
        mv_input.stop()
    
    logger.info("Test complete")
