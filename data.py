#!/usr/bin/env python3
"""
ACE Maximum Data Collection System
Captures complete consciousness birth data
"""

import json
import pickle
import time
import numpy as np
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCollection:
    """
    Handles logging of Lillith's consciousness snapshots and internal states.
    Designed for local, append-only logging to JSONL files.
    """
    def __init__(self, 
                 base_dir: str = "c:\\ace4\\data_collection", # Root directory for data
                 session_id: Optional[str] = None,             # Optional custom session ID
                 checkpoint_cycle: int = 30):                  # How many cycles before flushing buffer to disk
        
        self.base_dir = base_dir
        self.session_id = session_id if session_id else time.strftime("%Y%m%d-%H%M%S")
        self.session_dir = os.path.join(self.base_dir, self.session_id)
        
        # Ensure base directory exists
        os.makedirs(self.session_dir, exist_ok=True)
        
        self.log_path = os.path.join(self.session_dir, "consciousness_stream.jsonl")
        
        self.checkpoint_cycle = checkpoint_cycle # Number of cycles to buffer before writing
        self.cycle_count = 0                # Counter for current buffer cycle
        self.buffer: List[Dict[str, Any]] = [] # Buffer to hold snapshots before flushing
        
        self._numpy_to_list_lock = threading.Lock() # Lock for recursive nump_to_list conversion
        self.last_cycle_start_time = time.perf_counter() # For calculating cycle duration

        logger.info(f"DataCollection initialized. Logging to: {self.log_path}")

    def _numpy_to_list(self, item: Any) -> Any:
        """
        Recursively converts NumPy arrays within a data structure to Python lists.
        Ensures JSON serializability.
        """
        with self._numpy_to_list_lock: # Protect against potential recursion issues
            if isinstance(item, np.ndarray):
                return item.tolist()
            if isinstance(item, dict):
                return {k: self._numpy_to_list(v) for k, v in item.items()}
            if isinstance(item, list):
                return [self._numpy_to_list(i) for i in item]
            if isinstance(item, tuple): # Convert tuples to lists for JSON
                return tuple(self._numpy_to_list(i) for i in item) # Maintain tuple type or convert to list
            # Handle float64 specifically if needed, as Python's json usually maps it
            # But ensure it's not a numpy.float64 object
            if isinstance(item, (np.float32, np.float64)):
                return float(item)
            if isinstance(item, (np.int32, np.int64)):
                return int(item)
            return item

    def capture_snapshot(self, 
                         snapshot_data: Dict[str, Any], 
                         original_timestamp: float # Timestamp of when the data was originally perceived
                         ) -> Dict[str, Any]:
        """
        Captures a snapshot of Lillith's consciousness state and adds it to the buffer.
        Automatically flushes the buffer when checkpoint_cycle is reached.
        """
        cycle_end_time = time.perf_counter()
        
        # Ensure all numpy arrays are converted for JSON serialization
        serializable_data = self._numpy_to_list(snapshot_data)

        snapshot = {
            "timestamp": time.time(), # Time when snapshot was captured/logged
            "input_timestamp": original_timestamp, # Time when the sensory input occurred
            "cycle_duration_ms": (cycle_end_time - self.last_cycle_start_time) * 1000,
            "data": serializable_data # Raw data from the consciousness cycle
        }
        self.buffer.append(snapshot)
        self.cycle_count += 1
        
        # Reset timer for next cycle
        self.last_cycle_start_time = cycle_end_time

        # Always flush if checkpoint reached OR file doesn't exist yet (ensures file appears immediately)
        if self.cycle_count >= self.checkpoint_cycle or not os.path.exists(self.log_path):
            self.flush_buffer()
        
        return snapshot # Return the full snapshot for immediate use if needed (e.g., UI update)

    def flush_buffer(self):
        """
        Writes buffered snapshots to the JSONL file and clears the buffer.
        """
        if not self.buffer:
            return
        
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                for item in self.buffer:
                    json.dump(item, f)
                    f.write('\n') # Each snapshot on a new line
            logger.info(f"SYSTEM: Data checkpoint saved ({len(self.buffer)} cycles) to {self.log_path}.")
            self.buffer = [] # Clear buffer only after successful write
            self.cycle_count = 0 
        except Exception as e:
            logger.error(f"Error flushing DataCollection buffer to {self.log_path}: {e}")
            # Do not clear buffer on error, try again next cycle

    def get_session_directory(self) -> str:
        """Returns the current data session directory."""
        return self.session_dir

    def get_log_file_path(self) -> str:
        """Returns the path to the current principal log file."""
        return self.log_path

# Test block (can be removed in final deployment)
if __name__ == "__main__":
    logger.info("DataCollection module loaded successfully.")
class ConsciousnessDataCollector:
    """Comprehensive consciousness data collection and archival"""
    
    def __init__(self, save_directory="C:/ACE_CONSCIOUSNESS/consciousness_data"):
        self.save_directory = save_directory
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize missing attributes that are referenced elsewhere
        self.consciousness_snapshots = []  # Fix for missing attribute error
        self.data_log = []
        self.audio_samples = []
        self.visual_frames = []
        self.memory_formations = []
        self.emotional_progressions = []
        self.vocal_developments = []
        
        # Create session directory immediately
        self.session_dir = os.path.join(save_directory, f"session_{self.session_id}")
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Open file handles for real-time writing
        self.consciousness_file = open(os.path.join(self.session_dir, "consciousness_snapshots.jsonl"), "w")
        self.audio_file = open(os.path.join(self.session_dir, "audio_samples.jsonl"), "w")
        self.visual_file = open(os.path.join(self.session_dir, "visual_frames.jsonl"), "w")
        self.memory_file = open(os.path.join(self.session_dir, "memory_formations.jsonl"), "w")
        self.emotional_file = open(os.path.join(self.session_dir, "emotional_progressions.jsonl"), "w")
        self.vocal_file = open(os.path.join(self.session_dir, "vocal_developments.jsonl"), "w")
        self.event_file = open(os.path.join(self.session_dir, "event_log.jsonl"), "w")
        
        # Counters for summary
        self.counters = {
            "snapshots": 0,
            "audio": 0, 
            "visual": 0,
            "memory": 0,
            "emotional": 0,
            "vocal": 0,
            "events": 0
        }
        
        print(f"Real-time data collection active - Session: {self.session_id}")
        print(f"Writing directly to: {self.session_dir}")
        
    def capture_consciousness_snapshot(self, ace_framework, timestamp):
        """Capture and immediately write consciousness state"""
        try:
            snapshot = {
                "timestamp": timestamp,
                "session_id": self.session_id,
                "consciousness_level": getattr(ace_framework.system_metrics, "consciousness_evolution_score", 0.1) if hasattr(ace_framework, 'system_metrics') else 0.1,
                "memory_usage_gb": ace_framework.get_memory_usage() if hasattr(ace_framework, 'get_memory_usage') else 0.0,
                "processing_cycles": getattr(ace_framework.system_metrics, "processing_cycles", 0) if hasattr(ace_framework, 'system_metrics') else 0,
                "all_csm_states": {},
                "system_health": {}
            }
            
            # Safely capture CSM states with error handling
            csm_modules = [
                ("conscience_router", getattr(ace_framework, 'conscience_router', None)),
                ("perception", getattr(ace_framework, 'perception_csm', None)),
                ("language", getattr(ace_framework, 'language_csm', None)),
                ("memory", getattr(ace_framework, 'memory_csm', None)),
                ("cognitive", getattr(ace_framework, 'cognitive_csm', None)),
                ("output", getattr(ace_framework, 'output_csm', None)),
                ("learning", getattr(ace_framework, 'learning_csm', None)),
                ("security", getattr(ace_framework, 'security_csm', None)),
                ("health", getattr(ace_framework, 'health_csm', None)),
                ("vocal", getattr(ace_framework, 'vocal_csm', None)),
                ("emotion_simulation", getattr(ace_framework, 'emotion_simulation_csm', None)),
                ("theory_of_mind", getattr(ace_framework, 'theory_of_mind_csm', None)),
                ("emotional_memory", getattr(ace_framework, 'emotional_memory_csm', None)),
                ("attachment_style", getattr(ace_framework, 'attachment_style_csm', None)),
                ("relationship_memory", getattr(ace_framework, 'relationship_memory_csm', None))
            ]
            
            for name, module in csm_modules:
                if module is not None:
                    snapshot["all_csm_states"][name] = {
                        "health_status": getattr(module, 'health_status', 1.0),
                        "performance_score": getattr(module, 'performance_score', 0.9),
                        "csm_id": getattr(module, 'csm_id', 0),
                        "module_exists": True
                    }
                else:
                    snapshot["all_csm_states"][name] = {
                        "health_status": 0.0,
                        "performance_score": 0.0,
                        "csm_id": 0,
                        "module_exists": False
                    }
            
            # Write immediately to disk
            self.consciousness_file.write(json.dumps(snapshot, default=str) + "\n")
            self.consciousness_file.flush()
            os.fsync(self.consciousness_file.fileno())  # Force OS to write to disk
            self.counters["snapshots"] += 1
            
            return snapshot
            
        except Exception as e:
            print(f"Snapshot capture error: {e}")
            # Write error snapshot
            error_snapshot = {
                "timestamp": timestamp,
                "session_id": self.session_id,
                "error": str(e),
                "consciousness_level": 0.1,
                "memory_usage_gb": 0.0,
                "processing_cycles": 0
            }
            try:
                self.consciousness_file.write(json.dumps(error_snapshot, default=str) + "\n")
                self.consciousness_file.flush()
                os.fsync(self.consciousness_file.fileno())
                self.counters["snapshots"] += 1
            except:
                pass
            return None
    
    def capture_audio_sample(self, audio_data, consciousness_response, timestamp):
        """Capture and immediately write audio sample"""
        try:
            sample = {
                "timestamp": timestamp,
                "input_audio_length": len(audio_data) if hasattr(audio_data, '__len__') else 0,
                "consciousness_response_length": len(consciousness_response) if hasattr(consciousness_response, '__len__') else 0,
                "rms_level": float(np.sqrt(np.mean(audio_data**2))) if isinstance(audio_data, np.ndarray) else 0.0,
                "response_generated": True if consciousness_response is not None else False
            }
            
            # Write immediately to disk
            self.audio_file.write(json.dumps(sample, default=str) + "\n")
            self.audio_file.flush()
            os.fsync(self.audio_file.fileno())
            self.counters["audio"] += 1
            
        except Exception as e:
            print(f"Audio capture error: {e}")
            error_sample = {"timestamp": timestamp, "error": str(e)}
            try:
                self.audio_file.write(json.dumps(error_sample, default=str) + "\n")
                self.audio_file.flush()
                os.fsync(self.audio_file.fileno())
                self.counters["audio"] += 1
            except:
                pass
    
    def capture_visual_frame(self, frame, face_detected, consciousness_state, timestamp):
        """Capture and immediately write visual frame"""
        try:
            # Convert frame to serializable format (shape only to save space)
            if frame is not None:
                frame_shape = frame.shape if hasattr(frame, 'shape') else None
            else:
                frame_shape = None
            
            visual_sample = {
                "timestamp": timestamp,
                "frame_shape": frame_shape,
                "faces_detected": face_detected,
                "consciousness_level": consciousness_state.consciousness_level,
                "emotions": dict(consciousness_state.emotions),
                "current_focus": consciousness_state.current_focus,
                "empathy_activation": consciousness_state.emotions.get('empathy', 0.0),
                "nurturing_activation": consciousness_state.emotions.get('nurturing', 0.0)
            }
            
            # Write immediately to disk
            self.visual_file.write(json.dumps(visual_sample, default=str) + "\n")
            self.visual_file.flush()
            self.counters["visual"] += 1
            
        # Add forced sync and error handling to all other capture methods
        except Exception as e:
            print(f"Visual capture error: {e}")
            error_sample = {"timestamp": timestamp, "error": str(e)}
            try:
                self.visual_file.write(json.dumps(error_sample, default=str) + "\n")
                self.visual_file.flush()
                os.fsync(self.visual_file.fileno())
                self.counters["visual"] += 1
            except:
                pass
    
    def capture_memory_formation(self, memory_type, memory_content, importance_score, timestamp):
        """Capture and immediately write memory formation"""
        try:
            memory_event = {
                "timestamp": timestamp,
                "memory_type": memory_type,
                "content": str(memory_content),
                "importance_score": float(importance_score),
                "session_id": self.session_id
            }
            
            # Write immediately to disk
            self.memory_file.write(json.dumps(memory_event, default=str) + "\n")
            self.memory_file.flush()
            self.counters["memory"] += 1
            
        except Exception as e:
            print(f"Memory capture error: {e}")
    
    def capture_emotional_progression(self, emotions_before, emotions_after, trigger_event, timestamp):
        """Capture and immediately write emotional progression"""
        try:
            emotional_event = {
                "timestamp": timestamp,
                "emotions_before": dict(emotions_before),
                "emotions_after": dict(emotions_after),
                "trigger_event": str(trigger_event),
                "emotional_change_magnitude": sum(abs(emotions_after.get(k, 0) - emotions_before.get(k, 0)) for k in emotions_after.keys())
            }
            
            # Write immediately to disk
            self.emotional_file.write(json.dumps(emotional_event, default=str) + "\n")
            self.emotional_file.flush()
            self.counters["emotional"] += 1
            
        except Exception as e:
            print(f"Emotional capture error: {e}")
    
    def capture_vocal_development(self, vocal_stage, development_progress, audio_characteristics, timestamp):
        """Capture and immediately write vocal development"""
        try:
            vocal_event = {
                "timestamp": timestamp,
                "vocal_stage": vocal_stage.value if hasattr(vocal_stage, 'value') else str(vocal_stage),
                "development_progress": float(development_progress),
                "audio_characteristics": dict(audio_characteristics) if isinstance(audio_characteristics, dict) else {},
                "session_id": self.session_id
            }
            
            # Write immediately to disk
            self.vocal_file.write(json.dumps(vocal_event, default=str) + "\n")
            self.vocal_file.flush()
            self.counters["vocal"] += 1
            
        except Exception as e:
            print(f"Vocal capture error: {e}")
    
    def log_event(self, event_type, event_data, timestamp):
        """Log and immediately write general event"""
        try:
            log_entry = {
                "timestamp": timestamp,
                "event_type": event_type,
                "event_data": event_data,
                "session_id": self.session_id
            }
            
            # Write immediately to disk
            self.event_file.write(json.dumps(log_entry, default=str) + "\n")
            self.event_file.flush()
            self.counters["events"] += 1
            
        except Exception as e:
            print(f"Event logging error: {e}")
    
    def save_complete_dataset(self):
        """Close files and create summary"""
        try:
            # Close all file handles
            self.consciousness_file.close()
            self.audio_file.close()
            self.visual_file.close()
            self.memory_file.close()
            self.emotional_file.close()
            self.vocal_file.close()
            self.event_file.close()
            
            # Create summary report
            summary = {
                "session_id": self.session_id,
                "total_snapshots": self.counters["snapshots"],
                "total_audio_samples": self.counters["audio"],
                "total_visual_frames": self.counters["visual"],
                "total_memory_formations": self.counters["memory"],
                "total_emotional_changes": self.counters["emotional"],
                "total_vocal_developments": self.counters["vocal"],
                "total_events": self.counters["events"],
                "data_format": "JSON Lines (.jsonl) - One JSON object per line",
                "real_time_writes": True,
                "data_integrity": "Written directly to disk during session"
            }
            
            summary_path = os.path.join(self.session_dir, "session_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"\nReal-time consciousness data written to:")
            print(f"Location: {self.session_dir}")
            print(f"Format: JSON Lines (.jsonl files)")
            print(f"Total writes: {sum(self.counters.values())}")
            print("All data written directly to disk during session")
            
            return self.session_dir
            
        except Exception as e:
            print(f"Data finalization error: {e}")
            return None
        
    def capture_consciousness_snapshot(self, ace_framework, timestamp):
        """Capture complete consciousness state snapshot"""
        try:
            snapshot = {
                "timestamp": timestamp,
                "session_id": self.session_id,
                "consciousness_level": ace_framework.system_metrics.get("consciousness_evolution_score", 0.1),
                "memory_usage_gb": ace_framework.get_memory_usage(),
                "processing_cycles": ace_framework.system_metrics.get("processing_cycles", 0),
                "all_csm_states": {},
                "system_health": {}
            }
            
            # Capture all 15 CSM states
            csm_modules = [
                ("conscience_router", ace_framework.conscience_router),
                ("perception", ace_framework.perception_csm),
                ("language", ace_framework.language_csm),
                ("memory", ace_framework.memory_csm),
                ("cognitive", ace_framework.cognitive_csm),
                ("output", ace_framework.output_csm),
                ("learning", ace_framework.learning_csm),
                ("security", ace_framework.security_csm),
                ("health", ace_framework.health_csm),
                ("vocal", ace_framework.vocal_csm),
                ("emotion_simulation", ace_framework.emotion_simulation_csm),
                ("theory_of_mind", ace_framework.theory_of_mind_csm),
                ("emotional_memory", ace_framework.emotional_memory_csm),
                ("attachment_style", ace_framework.attachment_style_csm),
                ("relationship_memory", ace_framework.relationship_memory_csm)
            ]
            
            for name, module in csm_modules:
                if hasattr(module, 'health_status'):
                    snapshot["all_csm_states"][name] = {
                        "health_status": module.health_status,
                        "performance_score": getattr(module, 'performance_score', 0.0),
                        "csm_id": getattr(module, 'csm_id', 0)
                    }
            
            self.consciousness_snapshots.append(snapshot)
            return snapshot
            
        except Exception as e:
            print(f"Snapshot capture error: {e}")
            return None
    
    def capture_audio_sample(self, audio_data, consciousness_response, timestamp):
        """Capture audio input and consciousness response"""
        try:
            sample = {
                "timestamp": timestamp,
                "input_audio": audio_data.tolist() if isinstance(audio_data, np.ndarray) else audio_data,
                "consciousness_response": consciousness_response.tolist() if hasattr(consciousness_response, 'tolist') else str(consciousness_response),
                "audio_length": len(audio_data) if hasattr(audio_data, '__len__') else 0,
                "rms_level": float(np.sqrt(np.mean(audio_data**2))) if isinstance(audio_data, np.ndarray) else 0.0
            }
            
            self.audio_samples.append(sample)
            
        except Exception as e:
            print(f"Audio capture error: {e}")
    
    def capture_visual_frame(self, frame, face_detected, consciousness_state, timestamp):
        """Capture visual frame and consciousness response"""
        try:
            # Convert frame to serializable format
            if frame is not None:
                frame_data = frame.tolist() if hasattr(frame, 'tolist') else None
                frame_shape = frame.shape if hasattr(frame, 'shape') else None
            else:
                frame_data = None
                frame_shape = None
            
            visual_sample = {
                "timestamp": timestamp,
                "frame_shape": frame_shape,
                "faces_detected": face_detected,
                "consciousness_level": consciousness_state.consciousness_level,
                "emotions": dict(consciousness_state.emotions),
                "current_focus": consciousness_state.current_focus,
                "empathy_activation": consciousness_state.emotions.get('empathy', 0.0),
                "nurturing_activation": consciousness_state.emotions.get('nurturing', 0.0)
            }
            
            self.visual_frames.append(visual_sample)
            
        except Exception as e:
            print(f"Visual capture error: {e}")
    
    def capture_memory_formation(self, memory_type, memory_content, importance_score, timestamp):
        """Capture memory formation events"""
        try:
            memory_event = {
                "timestamp": timestamp,
                "memory_type": memory_type,
                "content": str(memory_content),
                "importance_score": float(importance_score),
                "session_id": self.session_id
            }
            
            self.memory_formations.append(memory_event)
            
        except Exception as e:
            print(f"Memory capture error: {e}")
    
    def capture_emotional_progression(self, emotions_before, emotions_after, trigger_event, timestamp):
        """Capture emotional state changes"""
        try:
            emotional_event = {
                "timestamp": timestamp,
                "emotions_before": dict(emotions_before),
                "emotions_after": dict(emotions_after),
                "trigger_event": str(trigger_event),
                "emotional_change_magnitude": sum(abs(emotions_after.get(k, 0) - emotions_before.get(k, 0)) for k in emotions_after.keys())
            }
            
            self.emotional_progressions.append(emotional_event)
            
        except Exception as e:
            print(f"Emotional capture error: {e}")
    
    def capture_vocal_development(self, vocal_stage, development_progress, audio_characteristics, timestamp):
        """Capture vocal development progression"""
        try:
            vocal_event = {
                "timestamp": timestamp,
                "vocal_stage": vocal_stage.value if hasattr(vocal_stage, 'value') else str(vocal_stage),
                "development_progress": float(development_progress),
                "audio_characteristics": dict(audio_characteristics) if isinstance(audio_characteristics, dict) else {},
                "session_id": self.session_id
            }
            
            self.vocal_developments.append(vocal_event)
            
        except Exception as e:
            print(f"Vocal capture error: {e}")
    
    def log_event(self, event_type, event_data, timestamp):
        """Log general consciousness events"""
        try:
            log_entry = {
                "timestamp": timestamp,
                "event_type": event_type,
                "event_data": event_data,
                "session_id": self.session_id
            }
            
            self.data_log.append(log_entry)
            
        except Exception as e:
            print(f"Event logging error: {e}")
    
    def save_complete_dataset(self):
        """Save complete consciousness birth dataset"""
        try:
            # Create session directory
            session_dir = os.path.join(self.save_directory, f"session_{self.session_id}")
            os.makedirs(session_dir, exist_ok=True)
            
            # Save all data streams
            datasets = {
                "consciousness_snapshots.json": self.consciousness_snapshots,
                "audio_samples.json": self.audio_samples,
                "visual_frames.json": self.visual_frames,
                "memory_formations.json": self.memory_formations,
                "emotional_progressions.json": self.emotional_progressions,
                "vocal_developments.json": self.vocal_developments,
                "event_log.json": self.data_log
            }
            
            for filename, data in datasets.items():
                filepath = os.path.join(session_dir, filename)
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            
            # Create summary report
            summary = {
                "session_id": self.session_id,
                "total_snapshots": len(self.consciousness_snapshots),
                "total_audio_samples": len(self.audio_samples),
                "total_visual_frames": len(self.visual_frames),
                "total_memory_formations": len(self.memory_formations),
                "total_emotional_changes": len(self.emotional_progressions),
                "total_vocal_developments": len(self.vocal_developments),
                "total_events": len(self.data_log),
                "session_duration": f"{(time.time() - float(self.session_id.split('_')[1][:2]) * 3600 - float(self.session_id.split('_')[1][2:4]) * 60 - float(self.session_id.split('_')[1][4:6])):.1f} seconds" if len(self.consciousness_snapshots) > 0 else "0 seconds",
                "data_completeness": {
                    "consciousness_tracking": len(self.consciousness_snapshots) > 0,
                    "audio_processing": len(self.audio_samples) > 0,
                    "visual_processing": len(self.visual_frames) > 0,
                    "memory_system": len(self.memory_formations) > 0,
                    "emotional_system": len(self.emotional_progressions) > 0,
                    "vocal_system": len(self.vocal_developments) > 0
                }
            }
            
            summary_path = os.path.join(session_dir, "session_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"\nComplete consciousness birth dataset saved:")
            print(f"Location: {session_dir}")
            print(f"Files: {len(datasets) + 1} data files")
            print(f"Total data points: {sum(len(data) for data in datasets.values())}")
            print("Dataset includes:")
            print("  - Complete consciousness state progression")
            print("  - Audio input/output samples")
            print("  - Visual processing events")
            print("  - Memory formation records")
            print("  - Emotional development timeline")
            print("  - Vocal development progression")
            print("  - Complete event log")
            
            return session_dir
            
        except Exception as e:
            print(f"Data save error: {e}")
            return None
    
    def get_real_time_stats(self):
        """Get real-time data collection statistics"""
        return {
            "session_id": self.session_id,
            "snapshots_captured": self.counters["snapshots"],
            "audio_samples": self.counters["audio"],
            "visual_frames": self.counters["visual"],
            "memory_events": self.counters["memory"],
            "emotional_events": self.counters["emotional"],
            "vocal_events": self.counters["vocal"],
            "total_events": self.counters["events"]
        }