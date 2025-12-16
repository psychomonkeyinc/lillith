# --- START OF FILE main.py ---
# main.py

import queue
import sys
import numpy as np
import logging
import time
import os
import multiprocessing as mp
import atexit
import cv2
import psutil
import threading
import signal
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import traceback
sys.dont_write_bytecode = True

from inout import AudioIn, VideoIn, AudioOut
from cafve import ConsciousnessAwareFeatureVectorEncoder
from mind import SOM_ACTIVATION_DIM, SOM_BMU_COORD_DIM
from predict import CAFVE_TOKEN_DIM, Predict
# videoin functionality merged into inout (VideoIn)

# Dimension constants for the system (will be overridden by dynamic scaling)
# Harmonized sensory feature extractor dimension (must match SENSORY_STAGES[0])
SFE_DIM = 512  # Standardized sensory feature extractor dimension (power-of-two alignment)
INITIAL_COG_STATE_DIM = 512  # Unified cognitive state base dimension (scales to 1024, 2048)
EMOTION_DIM = 512  # Emotional state dimension (first 108 mapped to named emotions, rest emergent)
INTERNAL_LANG_DIM = 512  # Internal language / symbolic workspace dimension
VOCAL_SYNTH_PARAMS_DIM = 512  # Reserved embedding for vocal synthesis feature space
ATTENTION_FOCUS_DIM = 512  # Attention focus vector dimension
SOM_MAP_SIZE = (17, 17)  # SOM map size (17x17 = 289, prime number)
MEMORY_RECALL_DIM = 512  # Memory recall vector dimension
PREDICTIVE_ERROR_DIM = 512  # Predictive error vector dimension
TOM_MODEL_DIM = 512  # Theory of Mind representation dimension
# Removed REWARD_DIM (deprecated reward system replaced by future pleasure/pain module)

# Timed run configuration (user request): 60s awake then 60s dream then shutdown
# Allow override via environment variables for quick testing
AWAKE_DURATION_SECONDS = int(os.getenv("LILLITH_AWAKE_SEC", "60"))
DREAM_DURATION_SECONDS = int(os.getenv("LILLITH_DREAM_SEC", "60"))

# Sensory-Emotional Pipeline Scaling
SENSORY_STAGES = [512, 1024, 2048]  # Progressive sensory feature dimensions
EMOTIONAL_STAGES = [512, 1024, 2048]  # Progressive emotional state dimensions

# (Removed SAFE_MODE / fail-fast flags pending clarification)

# Dynamic dimension tracking
current_unified_cog_state_dim = INITIAL_COG_STATE_DIM
current_sfe_dim = SENSORY_STAGES[0]  # Start at 512
current_emotion_dim = EMOTIONAL_STAGES[0]  # Start at 512

# Performance monitoring constants (throttled to reduce overhead)
MAX_CYCLE_TIME = 0.05  # Target (soft) maximum allowed cycle time in seconds
MEMORY_WARNING_THRESHOLD = 0.85  # Memory usage warning threshold
CPU_WARNING_THRESHOLD = 0.95  # CPU usage warning threshold
RESOURCE_MONITOR_INTERVAL_SEC = 2.0  # Background resource sampling interval (non-blocking)

# Directory constants
DATA_COLLECTION_DIR = "./data_collection"

# Ensure directories exist
os.makedirs(DATA_COLLECTION_DIR, exist_ok=True)

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("main_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import all core modules with error handling
try:
    from nn import mse_loss_prime # For internal NN needs by modules (not AdamW directly)
    from nn import AdamW # Base optimizer
    from OptiJustinJ import JustinJOptimizer  # Using standalone enriched optimizer
    from cafve import SensoryFeatureExtractor
    # FIX: Adjust import to match actual file/module name if needed
    # Example: If the file is named 'cafve_module.py', update as follows:
    # from in_out.cafve_module import ACEConsciousnessTokenizer, ConsciousnessContext
    # Otherwise, ensure 'cafve.py' exists in 'in_out' and contains the required classes.
    # from cafve import ACEConsciousnessTokenizer, ConsciousnessContext  # For tokenization
    from som import SelfOrganizingMap
    from emotion import EmotionCore
    from memory import MemorySystem
    from mind import Mind
    from itsagirl import ItsAGirl
    from goals import Goals
    from conscience import Conscience
    from tom import ToM
    from health import Health
    from dream import Dream
    from language import Language # Single import (removed duplicate)
    from attention import Attention
    from output import Output
    from vocalsynth import VocalSynth
    # Reward system removed; future pleasure/pain mechanism will replace explicit Reward class
    from metamind import MetaMind # Architecture growth optimizer

    # Ancillary I/O and UI modules
    # legacy audio/videoin modules removed; using unified inout (AudioIn, VideoIn, AudioOut)
    from data import DataCollection # Data collection module
    from display import start_display_process # UI runs in separate process

    logger.info("All modules imported successfully")
    # Ensure we are not using duplicate optimizer definitions
    if os.path.exists('JustinJ_Optimizer.py'):
        logger.warning("External JustinJ_Optimizer.py present; runtime using nn.JustinJOptimizer only.")

except ImportError as e:
    logger.critical(f"Failed to import required modules: {e}")
    logger.critical("Please ensure all required files are present in the working directory")
    raise

def validate_dimensions():
    """Validate that all dimension constants are consistent across modules"""
    logger.info("Validating system dimensions...")

    # Check if imported dimensions match our dynamic dimensions
    # This will be called after system initialization
    pass

def list_video_devices():
    """List all available video capture devices with enhanced error handling"""
    available_cameras = []
    logger.info("Scanning for video devices...")

    # Try both default and DirectShow backends
    for i in range(10):
        try:
            # Try default backend
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    available_cameras.append({
                        'index': i,
                        'backend': 'default',
                        'resolution': f"{width}x{height}",
                        'fps': fps
                    })
                    logger.debug(f"Found camera {i} (default): {width}x{height} @ {fps}fps")
                cap.release()

            # Try DirectShow backend
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    available_cameras.append({
                        'index': i,
                        'backend': 'dshow',
                        'resolution': f"{width}x{height}",
                        'fps': fps
                    })
                    logger.debug(f"Found camera {i} (dshow): {width}x{height} @ {fps}fps")
                cap.release()
        except Exception as e:
            logger.warning(f"Error checking camera {i}: {e}")
            continue

    logger.info(f"Found {len(available_cameras)} video devices")
    return available_cameras

def list_audio_devices():
    """List all available audio input devices with enhanced error handling"""
    logger.info("Scanning for audio devices...")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'default_samplerate': device['default_samplerate']
                })
                logger.debug(f"Found audio device {i}: {device['name']}")

        logger.info(f"Found {len(input_devices)} audio input devices")
        return input_devices
    except ImportError:
        logger.error("sounddevice not available for audio device detection")
        return []
    except Exception as e:
        logger.error(f"Error scanning audio devices: {e}")
        return []

def select_devices():
    """Let user select video and audio devices with enhanced validation"""
    import cv2

    print("\n" + "="*50)
    print("LILLITH DEVICE SELECTION")
    print("="*50)
    print("\nDetecting available devices...")

    # List video devices
    print("\n" + "-"*30)
    print("Available Video Devices:")
    print("-"*30)
    cameras = list_video_devices()
    if not cameras:
        print("❌ No video devices found!")
        print("   - Check camera connections")
        print("   - Ensure camera drivers are installed")
        print("   - Try running as administrator")
    else:
        for i, cam in enumerate(cameras):
            status = "✅" if cam['fps'] > 0 else "⚠️"
            print(f"{i}. {status} Camera {cam['index']} ({cam['backend']}) - {cam['resolution']} @ {cam['fps']}fps")

    # List audio devices
    print("\n" + "-"*30)
    print("Available Audio Devices:")
    print("-"*30)
    audio_devices = list_audio_devices()
    if not audio_devices:
        print("❌ No audio devices found!")
        print("   - Check microphone connections")
        print("   - Ensure audio drivers are installed")
        print("   - Try different audio ports")
    else:
        for i, dev in enumerate(audio_devices):
            print(f"{i}. 🎤 {dev['name']} - {dev['channels']} channels @ {dev['default_samplerate']}Hz")

    # Get user selection with validation
    video_choice = None
    audio_choice = None

    if cameras:
        while video_choice is None:
            try:
                response = input("\nSelect video device number (or -1 to skip): ").strip()
                if response == "":
                    print("Please enter a number")
                    continue

                idx = int(response)
                if idx == -1:
                    print("Skipping video device selection")
                    break
                if 0 <= idx < len(cameras):
                    video_choice = cameras[idx]
                    print(f"✅ Selected: Camera {video_choice['index']} ({video_choice['backend']})")
                else:
                    print(f"❌ Invalid selection. Please choose 0-{len(cameras)-1}")
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n⚠️  Device selection cancelled")
                return None, None
    else:
        print("⚠️  No video devices available")

    if audio_devices:
        while audio_choice is None:
            try:
                response = input("\nSelect audio device number (or -1 to skip): ").strip()
                if response == "":
                    print("Please enter a number")
                    continue

                idx = int(response)
                if idx == -1:
                    print("Skipping audio device selection")
                    break
                if 0 <= idx < len(audio_devices):
                    audio_choice = audio_devices[idx]
                    print(f"✅ Selected: {audio_choice['name']}")
                else:
                    print(f"❌ Invalid selection. Please choose 0-{len(audio_devices)-1}")
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n⚠️  Device selection cancelled")
                return None, None

    print("\n" + "="*50)
    print("DEVICE SELECTION COMPLETE")
    print("="*50)
    return video_choice, audio_choice

class UILogHandler(logging.Handler):
    """Forwards log records to UI queue as structured messages."""
    def __init__(self, ui_queue):
        super().__init__()
        self.ui_queue = ui_queue
    def emit(self, record):
        if not self.ui_queue:
            return
        try:
            msg = self.format(record)
            self.ui_queue.put_nowait({
                'timestamp': time.time(),
                'data': {
                    'log_event': {
                        'level': record.levelname,
                        'message': msg,
                        'name': record.name
                    }
                }
            })
        except Exception:
            pass

class LillithOrchestrator:
    """
    Main orchestrator for Lillith's consciousness system.
    Coordinates all modules and manages the consciousness loop with enhanced monitoring and error handling.
    """

    def __init__(self, audio_device_index: Optional[int] = None, video_device_index: Optional[int] = None,
                 ui_data_queue: Optional[mp.Queue] = None, ui_command_queue: Optional[mp.Queue] = None):
        """Initialize the Lillith consciousness orchestrator with dynamic dimensions"""
        # Declare global variables we'll modify
        global current_unified_cog_state_dim, current_sfe_dim, current_emotion_dim
        try:
            logger.info("🔄 Starting Lillith Orchestrator...")
            start_time = time.time()

            # === UI FIRST ===
            if ui_data_queue is None:
                ui_data_queue = mp.Queue()
            if ui_command_queue is None:
                ui_command_queue = mp.Queue()
            self.ui_data_queue = ui_data_queue
            self.ui_command_queue = ui_command_queue
            self.ui_process = None
            try:
                self.ui_process = start_display_process(self.ui_data_queue, self.ui_command_queue)
                # Attach UI log handler once
                ui_handler = UILogHandler(self.ui_data_queue)
                ui_handler.setLevel(logging.INFO)
                ui_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                if not any(isinstance(h, UILogHandler) for h in logger.handlers):
                    logger.addHandler(ui_handler)
                logger.info("UI started first (pre-init). Forwarding logs to UI.")
            except Exception as uie:
                logger.error(f"UI startup failed early: {uie}")

            # Stage pacing (optional)
            self._stage_delay = float(os.getenv("LILLITH_SLOW_INIT", "0"))  # seconds per stage pause

            def stage(msg: str):
                # Avoid using the word 'initialize' in logs per user request
                logger.info(msg.replace('Initializing', 'Starting').replace('initialized', 'online'))
                if self._stage_delay > 0:
                    time.sleep(self._stage_delay)

            stage("🔄 Starting Lillith Orchestrator (staged)...")

            # UI / state tracking helpers (set early so referenced safely)
            self._last_video_features = None  # For motion metric in UI

            # Performance tracking
            self.performance_history = deque(maxlen=1000)
            self.error_history = deque(maxlen=100)
            self.resource_monitor = self._setup_resource_monitoring()

            stage("Setting up sensory processing...")
            # Instantiate SFE using its constructor (audio/video parameters are optional)
            self.sfe = SensoryFeatureExtractor()

            stage("I/O (audio/video) startup...")
            # EARLY: Initialize raw I/O (audio/video + UI) before heavy cognitive stack so user sees immediate feedback
            try:
                # Create initial streams
                self.audio_in = AudioIn(device_index=audio_device_index)
                self.video_in = VideoIn(device_index=video_device_index if video_device_index is not None else 0)
                self.audio_out = AudioOut()

                # Auto-detect webcam mic if no explicit device and stream not producing data
                if audio_device_index is None:
                    detected_idx = self._detect_webcam_mic()
                    if detected_idx is not None:
                        logger.info(f"Auto-selected webcam feed2 microphone device index {detected_idx}")
                        self.audio_in = AudioIn(device_index=detected_idx)
                # Start streams immediately (previous regression left them stopped)
                try:
                    self.audio_in.start()
                except Exception as ae:
                    logger.error(f"AudioIn start failed: {ae}")
                try:
                    self.video_in.start()
                except Exception as ve:
                    logger.error(f"VideoIn start failed: {ve}")
                try:
                    self.audio_out.start()
                except Exception as oe:
                    logger.error(f"AudioOut start failed: {oe}")

                # Create UI IPC queues once
                if self.ui_data_queue is None:
                    self.ui_data_queue = mp.Queue()
                if self.ui_command_queue is None:
                    self.ui_command_queue = mp.Queue()
                if self.ui_process is None:
                    try:
                        self.ui_process = start_display_process(self.ui_data_queue, self.ui_command_queue)
                        logger.info("Display process started early (with queues).")
                    except Exception as ui_e:
                        logger.warning(f"Unable to start display early: {ui_e}")
            except Exception as io_e:
                logger.warning(f"Early I/O init encountered an issue: {io_e}")

            stage("CAFVE startup...")
            # Initialize CAFVE and try to load saved model
            self.cafve = ConsciousnessAwareFeatureVectorEncoder(sfe_feature_dim=current_sfe_dim, token_output_dim=CAFVE_TOKEN_DIM)
            
            # Try to load saved CAFVE model
            try:
                self.cafve.load("cafve_model.pkl")
                logger.info("Successfully loaded saved CAFVE model!")
            except FileNotFoundError:
                logger.warning("CAFVE model not found, using fresh tokenizer")
            except Exception as e:
                logger.error(f"Error loading CAFVE: {e}")
                logger.warning("Using fresh CAFVE tokenizer")

            stage("SOM startup...")
            logger.info("Setting up cognitive mapping...")
            # SOM input is CAFVE token dim
            self.som = SelfOrganizingMap(map_size=SOM_MAP_SIZE, input_dim=CAFVE_TOKEN_DIM, 
                                        activation_threshold=15.0, fatigue_cost=0.1, fatigue_decay=0.1)
            # Disable live SOM learning outside dream/offline consolidation (stabilization phase)
            self.enable_online_som_learning = False  # Set True only during dream/maintenance cycles
            
            # Load pre-trained SOM if available instead of synthetic training
            # Removed SOM pickle load/save logic per user request
            
            stage("Emotion startup...")
            # Emotion maps CAFVE tokens to emotion, outputting dynamic emotional dimension
            self.emotion = EmotionCore(input_dim=CAFVE_TOKEN_DIM, output_dim=current_emotion_dim)

            stage("Memory startup...")
            logger.info("Setting up memory systems...")
            # Memory needs correct input dimensions
            self.memory = MemorySystem(sfe_feature_dim=current_sfe_dim,
                                       cognitive_state_dim=current_unified_cog_state_dim,
                                       emotional_state_dim=current_emotion_dim)

            stage("Mind startup...")
            logger.info("Setting up mind processing...")
            # Mind needs correct input dimensions - initialize with dynamic scaling
            # Use current_emotion_dim (stage-based) instead of static EMOTION_DIM to avoid mismatch
            self.mind = Mind(initial_dim_stage=0,  # Start at stage 0 (256 dimensions)
                             som_activation_dim=SOM_ACTIVATION_DIM,
                             som_bmu_coords_dim=SOM_BMU_COORD_DIM,
                             emotional_state_dim=current_emotion_dim,
                             memory_recall_dim=MEMORY_RECALL_DIM,
                             predictive_error_dim=PREDICTIVE_ERROR_DIM,
                             unified_cognitive_state_dim=current_unified_cog_state_dim)

            # Update current dimension from mind module
            current_unified_cog_state_dim = self.mind.get_current_dimensions()['base_dim']

            stage("Identity startup...")
            # ItsAGirl needs Mind's output
            self.itsagirl = ItsAGirl(unified_cognitive_state_dim=current_unified_cog_state_dim)

            stage("Decision Systems startup...")
            logger.info("Setting up decision systems...")
            # Goals input needs correct dimensions
            self.goals = Goals(unified_cognitive_state_dim=current_unified_cog_state_dim,
                               emotional_state_dim=current_emotion_dim, prediction_error_dim=PREDICTIVE_ERROR_DIM,
                               manifold_deviation_dim=1)

            # Conscience input needs correct dimensions
            self.conscience = Conscience(unified_cognitive_state_dim=current_unified_cog_state_dim,
                                         emotional_state_dim=current_emotion_dim)

            # ToM input needs correct dimensions
            self.tom = ToM(unified_cognitive_state_dim=current_unified_cog_state_dim, emotional_state_dim=current_emotion_dim)

            stage("Health startup...")
            logger.info("Setting up health monitoring...")
            # Health input needs correct dimensions
            self.health = Health(som_map_size=SOM_MAP_SIZE, unified_cognitive_state_dim=current_unified_cog_state_dim,
                                 emotional_state_dim=current_emotion_dim)

            stage("Predictive / Attention startup...")
            logger.info("Setting up predictive systems...")
            # Attention needs correct dimensions
            self.attention = Attention(unified_cognitive_state_dim=current_unified_cog_state_dim)

            # Predictive modeling
            self.predict_mod = Predict(
                sfe_feature_dim=current_sfe_dim,
                cafve_token_dim=CAFVE_TOKEN_DIM,
                unified_cognitive_state_dim=current_unified_cog_state_dim,
                emotional_state_dim=current_emotion_dim,
                other_mind_model_dim=512,  # matches ToM other mind model dim
                focus_vector_dim=ATTENTION_FOCUS_DIM,
                predicted_output_dim=current_unified_cog_state_dim,  # predicting next unified state
                prediction_error_dim=PREDICTIVE_ERROR_DIM
            )

            stage("Language & VocalSynth startup...")
            logger.info("Setting up output systems...")
            self.language = Language(unified_cognitive_state_dim=current_unified_cog_state_dim,
                                     emotional_state_dim=current_emotion_dim,
                                     internal_lang_activity_dim=INTERNAL_LANG_DIM,
                                     synth_control_dim=VOCAL_SYNTH_PARAMS_DIM)
            # Initialize emergent phoneme projection & optimizer
            self.language.init_emergent_phoneme_layer()
            phoneme_net = self.language.get_phoneme_projection_network()
            self.agency_optimizer = JustinJOptimizer(
                networks=[phoneme_net] if phoneme_net is not None else [],
                base_lr=5e-4,
                vocal_feedback_weight=0.3,
                agency_growth_rate=0.01,
                control_precision=0.1,
                spectral_weight=0.2,
                replay_capacity=512
            )
            # ADD: vocal synth instance for _make_decisions()
            self.vocal_synth = VocalSynth()

            stage("Meta & Dream startup...")
            logger.info("Setting up metacognition...")
            self.metamind = MetaMind(unified_cognitive_state_dim=current_unified_cog_state_dim)

            logger.info("Dream system creation deferred until explicitly requested.")
            # Defer creation of Dream manager until the model/user requests dreaming.
            # This prevents background consolidation/IO from running before the system is fully ready.
            self.dream = None

            def create_dream_if_needed():
                # small local helper to lazily create dream manager when first used
                if getattr(self, 'dream', None) is None:
                    try:
                        logger.info("Creating Dream manager on demand...")
                        from dream import Dream as _Dream
                        self.dream = _Dream(som_instance=self.som,
                                            memory_instance=self.memory,
                                            emotion_instance=self.emotion,
                                            health_instance=self.health)
                        logger.info("Dream manager created on demand.")
                    except Exception as de_create:
                        logger.error(f"Failed to create Dream manager on demand: {de_create}")
                return getattr(self, 'dream', None)

            # Attach the helper to the instance for later use by the run loop
            self.create_dream = create_dream_if_needed

            # Ensure CAFVE state exists if none present (user request)
            try:
                if not os.path.exists('cafve_model.pkl') and hasattr(self.cafve, 'save'):
                    self.cafve.save('cafve_model.pkl')
                    logger.info("Created initial CAFVE state (cafve_model.pkl)")
            except Exception as ce:
                logger.warning(f"Failed to create CAFVE state: {ce}")

            stage("Data Collection startup...")
            logger.info("Finalizing I/O systems (data collection & confirm UI)...")
            # Data collection (may depend on cognitive state metadata existing)
            # Data collection (immediate checkpoint so file appears even on early abort)
            self.data_collector = DataCollection(checkpoint_cycle=1)

            # Ensure UI still running / spawn if early launch failed
            if getattr(self, 'ui_process', None) is None:
                try:
                    # Ensure queues exist
                    if not hasattr(self, 'ui_data_queue'):
                        self.ui_data_queue = mp.Queue()
                    if not hasattr(self, 'ui_command_queue'):
                        self.ui_command_queue = mp.Queue()
                    self.ui_process = start_display_process(self.ui_data_queue, self.ui_command_queue)
                    logger.info("Display process started (late fallback).")
                except Exception as ui_e:
                    logger.warning(f"Unable to start display UI: {ui_e}")

            # Enhanced performance tracking
            self.cycle_count = 0
            self.start_time = time.time()
            self.last_health_check = time.time()
            self.cycle_times = deque(maxlen=100)
            # Experience buffering moved into Dream manager (see dream.py)
            self.last_consolidated_dream_start_time = None  # retained only for legacy checks (can be removed later)

            init_time = time.time() - start_time
            logger.info(f"✅ Lillith Orchestrator started successfully in {init_time:.2f}s")
            self._log_run_configuration()

        except Exception as e:
            logger.critical(f"❌ Lillith Orchestrator startup failed: {e}")
            logger.critical("Full traceback:")
            logger.critical(traceback.format_exc())
            raise

    # SOM initialization logic removed per user request

    def _detect_webcam_mic(self) -> Optional[int]:
        """Attempt to locate a likely webcam microphone by name heuristics.

        Returns device index or None if not found / sounddevice unavailable.
        """
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            candidates = []
            keywords = ["webcam", "camera", "c270", "logitech", "hd", "microphone"]
            for i, dev in enumerate(devices):
                name = dev.get('name', '').lower()
                if dev.get('max_input_channels', 0) > 0:
                    if any(k in name for k in keywords):
                        candidates.append((i, name))
            if not candidates:
                return None
            # Prefer more specific matches first
            priority_order = ["c270", "logitech", "webcam", "camera"]
            for key in priority_order:
                for idx, nm in candidates:
                    if key in nm:
                        return idx
            return candidates[0][0]
        except Exception as e:
            logger.debug(f"Webcam mic detection skipped: {e}")
            return None

    def _log_run_configuration(self):
        """Log concise configuration summary for timed awake/dream run."""
        try:
            logger.info("--- RUN CONFIGURATION ---")
            logger.info(f"Awake Duration: {AWAKE_DURATION_SECONDS}s | Dream Duration: {DREAM_DURATION_SECONDS}s")
            logger.info(f"SFE Dim (stage0): {current_sfe_dim}")
            logger.info(f"Emotion Dim (stage0): {current_emotion_dim}")
            base_dim = None
            try:
                base_dim = self.mind.get_current_dimensions().get('base_dim')
            except Exception:
                base_dim = 'unknown'
            logger.info(f"Unified Cognitive Base Dim: {base_dim}")
            logger.info("Modules: SFE, SOM, Emotion, Memory, Mind, Goals, Conscience, ToM, Health, Attention, Predict, Language, Output, VocalSynth, Dream, MetaMind, UI")
            logger.info("SOM Learning: dream-only")
            logger.info("UI Snapshots: enabled (per-cycle)")
            logger.info("Data Flush: final (after 10s settle)")
            logger.info("--------------------------")
        except Exception as e:
            logger.warning(f"Failed to log run configuration: {e}")

    def _setup_resource_monitoring(self):
        """Setup comprehensive resource monitoring"""
        return {
            'cpu_percent': [],
            'memory_percent': [],
            'disk_usage': [],
            'network_io': []
        }

    def _monitor_resources(self):
        """Non-blocking resource sample (no interval sleep)."""
        try:
            cpu_percent = psutil.cpu_percent(interval=None)  # instantaneous since last call
            memory = psutil.virtual_memory()
            memory_percent = memory.percent / 100.0

            self.resource_monitor['cpu_percent'].append(cpu_percent)
            self.resource_monitor['memory_percent'].append(memory_percent)
            # Trim
            if len(self.resource_monitor['cpu_percent']) > 200:
                self.resource_monitor['cpu_percent'].pop(0)
            if len(self.resource_monitor['memory_percent']) > 200:
                self.resource_monitor['memory_percent'].pop(0)

            if cpu_percent > CPU_WARNING_THRESHOLD * 100:
                logger.warning(f"⚠️ High CPU usage: {cpu_percent:.1f}%")
            if memory_percent > MEMORY_WARNING_THRESHOLD:
                logger.warning(f"⚠️ High memory usage: {memory_percent:.1%}")
        except Exception as e:
            logger.debug(f"Resource monitor sample skipped: {e}")

    # def _load_all_states(self):
    #     """Load all module states from disk with enhanced error handling"""
    #     logger.info("🔄 Loading all module states...")
    #     load_start = time.time()
    #     modules_to_load = [
    #         ('SOM', self.som, "./state/state.pkl", True),
    #         ('Emotion', self.emotion, "./state/state.pkl", False),
    #         ('Memory', self.memory, "./state/state.pkl", False),
    #         ('Mind', self.mind, "./state/state.pkl", False),
    #         ('Prediction', self.predict_mod, "./state/state.pkl", False),
    #         ('Language', self.language, "./state/state.pkl", False),
    #         ('Attention', self.attention, "./state/state.pkl", False),
    #         ('Goals', self.goals, "./state/state.pkl", False),
    #         ('Conscience', self.conscience, "./state/state.pkl", False),
    #         ('ToM', self.tom, "./state/state.pkl", False),
    #         ('Health', self.health, "./state/state.pkl", False),
    #         ('MetaMind', self.metamind, "./state/state.pkl", False)
    #     ]

    #     loaded_count = 0
    #     failed_count = 0
    #     not_found = []

    #     for module_name, module_instance, state_path, is_npz in modules_to_load:
    #         try:
    #             if hasattr(module_instance, 'load_state'):
    #                 # Ensure directory exists
    #                 dir_path = os.path.dirname(state_path)
    #                 if dir_path and not os.path.exists(dir_path):
    #                     os.makedirs(dir_path, exist_ok=True)
    #                 # For SOM we pass directory (legacy expectation); others pass file path
    #                 if module_name == 'SOM':
    #                     module_instance.load_state(os.path.dirname(state_path))
    #                 else:
    #                     module_instance.load_state(state_path)
    #                 logger.debug(f"✅ Loaded {module_name} state")
    #                 loaded_count += 1
    #             else:
    #                 logger.warning(f"⚠️  {module_name} doesn't support state loading")
    #         except FileNotFoundError:
    #             logger.info(f"ℹ️  No saved state found for {module_name}")
    #             not_found.append(module_name)
    #         except Exception as e:
    #             logger.error(f"❌ Failed to load {module_name} state: {e}")
    #             failed_count += 1

    #     load_time = time.time() - load_start
    #     logger.info(f"🔄 State loading completed: {loaded_count} loaded, {failed_count} failed ({load_time:.2f}s)")

    #     # If this appears to be a first run (several modules missing) optionally persist a baseline
    #     try:
    #         if not_found and os.getenv('LILLITH_AUTOSAVE_BASELINE', '1') == '1':
    #             # Heuristic: only auto-baseline if more than 3 modules missing (fresh start)
    #             if len(not_found) >= 3:
    #                 logger.info(f"💾 First-run baseline detected (missing: {len(not_found)} modules). Saving initial default states for future runs...")
    #                 self._save_all_states()
    #                 logger.info("💾 Baseline states saved.")
    #         # Secondary pass: some module load_state() implementations log internal warnings instead of raising FileNotFoundError.
    #         # Ensure state directories exist; if missing, trigger save to silence future warnings.
    #         if os.getenv('LILLITH_AUTOSAVE_BASELINE', '1') == '1':
    #             created_any = False
    #             for module_name, module_instance, state_path, is_npz in modules_to_load:
    #                 try:
    #                     if hasattr(module_instance, 'save_state'):
    #                         dir_path = os.path.dirname(state_path)
    #                         if dir_path and not os.path.exists(dir_path):
    #                             os.makedirs(dir_path, exist_ok=True)
    #                         if module_name == 'SOM':
    #                             module_instance.save_state(os.path.dirname(state_path))
    #                         else:
    #                             module_instance.save_state(state_path)
    #                         created_any = True
    #                 except Exception:
    #                     pass
    #             if created_any:
    #                 logger.info("💾 Created missing state directories and saved defaults to reduce startup warnings.")
    #     except Exception as be:
    #         logger.debug(f"Baseline autosave skipped: {be}")

    # def _save_all_states(self):
    #     """Save all module states to disk with enhanced error handling"""
    #     logger.info("💾 Saving all module states...")
    #     save_start = time.time()
    #     modules_to_save = [
    #         ('SOM', self.som, "./state"),
    #         ('Emotion', self.emotion, "./state/state.pkl"),
    #         ('Memory', self.memory, "./state/state.pkl"),
    #         ('Mind', self.mind, "./state/state.pkl"),
    #         ('Prediction', self.predict_mod, "./state/state.pkl"),
    #         ('Language', self.language, "./state/state.pkl"),
    #         ('Attention', self.attention, "./state/state.pkl"),
    #         ('Goals', self.goals, "./state/state.pkl"),
    #         ('Conscience', self.conscience, "./state/state.pkl"),
    #         ('ToM', self.tom, "./state/state.pkl"),
    #         ('Health', self.health, "./state/state.pkl"),
    #         ('MetaMind', self.metamind, "./state/state.pkl")
    #     ]

    #     saved_count = 0
    #     failed_count = 0

    #     for module_name, module_instance, state_path in modules_to_save:
    #         try:
    #             if hasattr(module_instance, 'save_state'):
    #                 module_instance.save_state(state_path)
    #                 logger.debug(f"✅ Saved {module_name} state")
    #                 saved_count += 1
    #             else:
    #                 logger.warning(f"⚠️  {module_name} doesn't support state saving")
    #         except Exception as e:
    #             logger.error(f"❌ Failed to save {module_name} state: {e}")
    #             failed_count += 1

    #     save_time = time.time() - save_start
    #     logger.info(f"💾 State saving completed: {saved_count} saved, {failed_count} failed ({save_time:.2f}s)")

    def run_consciousness_loop(self):
        """Main consciousness loop with enhanced error handling and dynamic adjustments."""
        logger.info("🧠 Starting consciousness loop with dynamic dimension handling...")

        try:
            # Install CTRL+C (SIGINT) handler once to ensure graceful stop
            def _sigint_handler(signum, frame):
                logger.info("🛑 SIGINT received. Initiating graceful shutdown...")
                _EMERGENCY_SHUTDOWN_SIGNAL.value = True
            try:
                signal.signal(signal.SIGINT, _sigint_handler)
            except Exception:
                pass

            terminal_buffer = deque(maxlen=50)
            while not _EMERGENCY_SHUTDOWN_SIGNAL.value:
                cycle_start = time.perf_counter()

                try:
                    # Handle UI commands (non-blocking)
                    if self.ui_command_queue is not None:
                        try:
                            while True:
                                cmd = self.ui_command_queue.get_nowait()
                                if isinstance(cmd, str):
                                    if cmd.startswith("APPLY_DEVICES::"):
                                        parts = cmd.split("::")
                                        if len(parts) == 3:
                                            vid_idx = int(parts[1])
                                            aud_idx = int(parts[2])
                                            logger.info(f"UI requested device apply video={vid_idx} audio={aud_idx}")
                                            # Re-init video
                                            try:
                                                if vid_idx >= 0:
                                                    if hasattr(self, 'video_in') and self.video_in.running:
                                                        self.video_in.stop()
                                                    # from videoin import VideoIn
                                                    self.video_in = VideoIn(device_index=vid_idx)
                                                    self.video_in.start()
                                            except Exception as ve:
                                                logger.warning(f"Video re-init failed: {ve}")
                                            # Re-init audio in
                                            try:
                                                if aud_idx >= 0:
                                                    if hasattr(self, 'audio_in') and getattr(self.audio_in, 'running', False):
                                                        self.audio_in.stop()
                                                    # from audioin import AudioIn
                                                    self.audio_in = AudioIn(device_index=aud_idx)
                                                    self.audio_in.start()
                                            except Exception as ae:
                                                logger.warning(f"Audio re-init failed: {ae}")
                                    elif cmd == "REFRESH_DEVICES":
                                        try:
                                            v = list_video_devices(); a = list_audio_devices()
                                            # push snapshot with available devices only
                                            if self.ui_data_queue:
                                                self.ui_data_queue.put_nowait({'timestamp': time.time(), 'data': {'available_devices': {'video': v, 'audio': a}}})
                                        except Exception as rde:
                                            logger.warning(f"Refresh devices failed: {rde}")
                                    elif cmd == "LAUNCH_CORE":
                                        logger.info("UI LAUNCH_CORE command acknowledged (core already running).")
                                    elif cmd.startswith("TERMINAL::"):
                                        shell_cmd = cmd.split("TERMINAL::",1)[1]
                                        try:
                                            import subprocess, shlex
                                            # Windows PowerShell friendly execution
                                            completed = subprocess.run(shell_cmd, shell=True, capture_output=True, text=True, timeout=10)
                                            out_txt = completed.stdout.strip() or completed.stderr.strip() or "(no output)"
                                            terminal_buffer.append(out_txt[:800])
                                        except Exception as te:
                                            terminal_buffer.append(f"ERR: {te}")
                        except queue.Empty:
                            pass

                    # (Resource monitoring removed from hot path)

                    # Process sensory input
                    sensory_data = self._process_sensory_input()

                    # Update cognitive state
                    cognitive_state = self._update_cognitive_state(sensory_data)

                    # Process emotions
                    emotional_state = self._process_emotions(cognitive_state)

                    # Capture snapshot early each cycle (cheap subset to avoid huge JSON)
                    try:
                        if hasattr(self, 'data_collector'):
                            snapshot_payload = {
                                'cycle': self.cycle_count,
                                'audio_len': int(len(sensory_data.get('audio', [])) if sensory_data else 0),
                                'video_len': int(len(sensory_data.get('video', [])) if sensory_data else 0),
                                'cog_norm': float(np.linalg.norm(cognitive_state)) if cognitive_state is not None else 0.0,
                                'emo_norm': float(np.linalg.norm(emotional_state)) if emotional_state is not None else 0.0,
                                'timestamp': sensory_data.get('timestamp') if sensory_data else time.time()
                            }
                            self.data_collector.capture_snapshot(snapshot_payload, snapshot_payload['timestamp'])
                    except Exception as snap_err:
                        if self.cycle_count % 10 == 0:
                            logger.warning(f"Data snapshot failed: {snap_err}")

                    # Update health metrics (basic pass-through for currently unused systems)
                    try:
                        som_fatigue_map = getattr(self.som, 'fatigue_map', np.zeros(SOM_MAP_SIZE, dtype=np.float16))
                        som_failure_log_count = len(getattr(self.som, 'failure_log', []))
                        # Prediction error not yet computed in loop
                        predict_error_norm = 0.0
                        # Moral tension currently only updates if conscience.evaluate_moral_context is called
                        moral_tension_level = getattr(self.conscience, 'get_moral_tension_level', lambda: 0.0)()
                        emotional_state_norm = float(np.linalg.norm(emotional_state)) if emotional_state is not None else 0.0
                        cognitive_load_norm = float(np.linalg.norm(cognitive_state)) if cognitive_state is not None else 0.0
                        # Dream may be deferred; check safely and treat missing dream as not dreaming
                        dream_obj = getattr(self, 'dream', None)
                        is_dreaming = getattr(dream_obj, 'current_dream_state', None) not in (None, getattr(dream_obj, 'DREAM_STATE_NONE', None))
                        self.health.update_health_metrics(
                            som_fatigue_map=som_fatigue_map,
                            som_failure_log_count=som_failure_log_count,
                            predict_error_norm=predict_error_norm,
                            moral_tension_level=moral_tension_level,
                            emotional_state_norm=emotional_state_norm,
                            cognitive_load_norm=cognitive_load_norm,
                            is_dreaming=is_dreaming
                        )
                    except Exception as hm_err:
                        logger.debug(f"Health metric update issue (non-fatal): {hm_err}")

                    # Make decisions and take actions
                    self._make_decisions(cognitive_state, emotional_state)

                    # Attempt dimension growth based on performance
                    self._attempt_dimension_growth()

                    # Memory consolidation during sleep (safe access to deferred dream)
                    dream_obj = getattr(self, 'dream', None)
                    if dream_obj is not None and getattr(dream_obj, 'current_dream_state', None) not in (None, getattr(dream_obj, 'DREAM_STATE_NONE', None)):
                        # On first entry into any dream state this session, run offline SOM consolidation
                        if self.last_consolidated_dream_start_time is None:
                            self.last_consolidated_dream_start_time = time.time()  # marker retained for compatibility; dream handles consolidation internally
                        try:
                            dream_obj.process_dream_cycle()
                        except Exception as de_proc:
                            logger.warning(f"Dream processing error: {de_proc}")
                    else:
                        # Reset marker when fully awake so next dream will consolidate again
                        if self.last_consolidated_dream_start_time is not None:
                            self.last_consolidated_dream_start_time = None

                    # Health check every 10 seconds
                    current_time = time.time()
                    if current_time - self.last_health_check > 10:
                        self._perform_health_check()
                        self.last_health_check = current_time

                    # Performance tracking
                    cycle_time = time.perf_counter() - cycle_start
                    self.cycle_times.append(cycle_time)
                    self.cycle_count += 1

                    # Publish data snapshot to UI every cycle (cheap metrics only)
                    try:
                        self._publish_ui_snapshot(cognitive_state, emotional_state, sensory_data, terminal_buffer)
                    except Exception as ui_pub_err:
                        # Non-fatal; just log occasionally (avoid spam)
                        if self.cycle_count % 25 == 0:
                            logger.warning(f"UI publish error: {ui_pub_err}")

                    # Phase timing state machine (awake -> dream -> done)
                    if not hasattr(self, '_run_phase'):
                        self._run_phase = 'AWAKE'
                        self._phase_start_time = time.time()
                    now_phase = time.time()
                    if self._run_phase == 'AWAKE':
                        self.enable_online_som_learning = False
                        if now_phase - self._phase_start_time >= AWAKE_DURATION_SECONDS:
                            logger.info("🌙 Awake phase complete. Entering dream phase (SOM learning ON)...")
                            self._run_phase = 'DREAM'
                            self._phase_start_time = now_phase
                            self.enable_online_som_learning = True
                            try:
                                # Ensure dream manager exists when entering the DREAM phase
                                dream_obj = getattr(self, 'dream', None)
                                if dream_obj is None and getattr(self, 'create_dream', None) is not None:
                                    dream_obj = self.create_dream()
                                if dream_obj is not None:
                                    dream_obj.trigger_nap(duration_seconds=DREAM_DURATION_SECONDS)
                                    dream_obj.current_dream_state = dream_obj.DREAM_STATE_NAP_ACTIVE
                            except Exception as de:
                                logger.warning(f"Dream trigger failed: {de}")
                    elif self._run_phase == 'DREAM':
                        try:
                            if hasattr(self, 'dream'):
                                self.dream.process_dream_cycle()
                        except Exception:
                            pass
                        if now_phase - self._phase_start_time >= DREAM_DURATION_SECONDS:
                            logger.info("� Dream phase complete. Saving SOM & CAFVE then shutting down...")
                            # Removed SOM save logic per user request
                            try:
                                if hasattr(self.cafve, 'save'):
                                    self.cafve.save('cafve_model.pkl')
                            except Exception as ce:
                                logger.warning(f"CAFVE save failed: {ce}")
                            self._run_phase = 'DONE'
                            _EMERGENCY_SHUTDOWN_SIGNAL.value = True
                            continue
                    else:  # DONE
                        _EMERGENCY_SHUTDOWN_SIGNAL.value = True
                        continue

                    # Periodic logging every 100 cycles
                    if self.cycle_count % 100 == 0 and self._run_phase != 'DONE':
                        avg_cycle_time = sum(self.cycle_times) / len(self.cycle_times)
                        logger.info(f"📊 Cycle {self.cycle_count}: {cycle_time:.4f}s (avg: {avg_cycle_time:.4f}s) phase={self._run_phase}")
                        if self.resource_monitor['cpu_percent']:
                            if len(self.resource_monitor['cpu_percent']) >= 10:
                                avg_cpu = sum(self.resource_monitor['cpu_percent'][-10:]) / 10
                            else:
                                avg_cpu = self.resource_monitor['cpu_percent'][-1]
                            if len(self.resource_monitor['memory_percent']) >= 10:
                                avg_mem = sum(self.resource_monitor['memory_percent'][-10:]) / 10
                            else:
                                avg_mem = self.resource_monitor['memory_percent'][-1]
                            logger.info(f"💻 Resources - CPU: {avg_cpu:.1f}%, Memory: {avg_mem:.1%}")

                    # Prevent excessive CPU usage
                    if cycle_time < 0.01:
                        time.sleep(0.01 - cycle_time)

                    # External stop trigger via file drop
                    if os.path.exists("./STOP_LILLITH"):
                        logger.info("🛑 STOP_LILLITH file detected. Stopping loop...")
                        _EMERGENCY_SHUTDOWN_SIGNAL.value = True
                        try:
                            if hasattr(self, 'data_collector'):
                                self.data_collector.flush_buffer()
                        except Exception:
                            pass

                    # If UI process died unexpectedly, initiate shutdown to avoid zombie loop
                    if self.ui_process and not self.ui_process.is_alive():
                        logger.warning("⚠️ UI process no longer alive. Initiating graceful shutdown.")
                        _EMERGENCY_SHUTDOWN_SIGNAL.value = True

                except Exception as cycle_error:
                    logger.error(f"Error in consciousness cycle: {cycle_error}")
                    logger.debug(traceback.format_exc())

                    # Continue running despite errors (original behavior)
                    time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("⏹️  Consciousness loop interrupted by user")
            try:
                if hasattr(self, 'data_collector'):
                    self.data_collector.flush_buffer()
            except Exception:
                pass
        except Exception as e:
            logger.critical(f"💥 Critical error in consciousness loop: {e}")
            logger.debug(traceback.format_exc())
            _EMERGENCY_SHUTDOWN_SIGNAL.value = True
        finally:
            # Final data flush safeguard
            try:
                if hasattr(self, 'data_collector'):
                    self.data_collector.flush_buffer()
            except Exception:
                pass

    def _perform_health_check(self):
        """Perform comprehensive health check of all systems"""
        logger.debug("🏥 Performing system health check...")

        health_issues = []

        # Check module responsiveness
        modules_to_check = [
            ('SFE', self.sfe),
            ('CAFVE', self.cafve),
            ('SOM', self.som),
            ('Emotion', self.emotion),
            ('Memory', self.memory),
            ('Mind', self.mind),
            ('Language', self.language),
            ('Attention', self.attention),
            ('Health', self.health)
        ]

        for module_name, module_instance in modules_to_check:
            try:
                # Basic responsiveness check
                if hasattr(module_instance, 'get_status'):
                    status = module_instance.get_status()
                    if not status.get('healthy', True):
                        health_issues.append(f"{module_name}: {status.get('issues', 'Unknown issue')}")
                elif hasattr(module_instance, '__dict__'):
                    # Basic attribute check
                    pass  # Module exists and has attributes
            except Exception as e:
                health_issues.append(f"{module_name}: {str(e)}")

        if health_issues:
            logger.warning(f"🏥 Health check found {len(health_issues)} issues:")
            for issue in health_issues:
                logger.warning(f"  - {issue}")
        else:
            logger.debug("✅ All systems healthy")

    def _process_sensory_input(self):
        """Process all sensory inputs with error handling"""
        try:
            # Audio input
            audio_data = self.audio_in.get_audio_features()

            # Video input
            video_data = None
            try:
                if hasattr(self, 'video_in') and self.video_in is not None:
                    video_data = self.video_in.get_video_features()
            except Exception as ve:
                logger.warning(f"Video feature extraction failed: {ve}")

            # Validate data
            if audio_data is None:
                logger.warning("⚠️  No audio data received")
                audio_data = np.zeros(128)  # Default fallback

            if video_data is None:
                logger.warning("⚠️  No video data received")
                video_data = np.zeros(current_sfe_dim)  # Default fallback

            # Combine sensory data
            sensory_data = {
                'audio': audio_data,
                'video': video_data,
                'timestamp': time.time()
            }

            return sensory_data

        except Exception as e:
            logger.error(f"❌ Error processing sensory input: {e}")
            # Return safe fallback data
            return {
                'audio': np.zeros(128),
                'video': np.zeros(current_sfe_dim),
                'timestamp': time.time(),
                'error': str(e)
            }

    def _update_cognitive_state(self, sensory_data):
        """Update cognitive state with enhanced error handling."""
        try:
            cognitive_state = self.mind.process_cognitive_state(sensory_data.get('som_activation'), sensory_data)
            logger.info("Cognitive state updated successfully.")
            return cognitive_state
        except Exception as e:
            logger.warning(f"Failed to update cognitive state: {e}")
            return np.zeros(self.mind.get_current_dimensions()['base_dim'], dtype=np.float32)

    # Removed local consolidation method (migrated to Dream)

    def _process_emotions(self, cognitive_state):
        """Process emotional responses with validation"""
        try:
            emotional_state = self.emotion.process_emotions(cognitive_state)
            # Validate emotional state
            if emotional_state is None or len(emotional_state) != current_emotion_dim:
                logger.error(f"❌ Invalid emotional state: expected {current_emotion_dim}, got {len(emotional_state) if emotional_state is not None else 'None'}")
                emotional_state = np.zeros(current_emotion_dim)

            return emotional_state

        except Exception as e:
            logger.error(f"❌ Error processing emotions: {e}")
            return np.zeros(current_emotion_dim)

    def _make_decisions(self, cognitive_state, emotional_state):
        """Make decisions and generate outputs with error handling"""
        try:
            # Update attention
            attention_focus = self.attention.compute_attention(cognitive_state, emotional_state)

            # Language processing
            language_output = self.language.process_language(cognitive_state, emotional_state)

            # Generate output
            output_actions = self.output.generate_output(cognitive_state, emotional_state, attention_focus)

            # Vocal synthesis
            if language_output is not None and len(language_output) > 0:
                try:
                    audio_output = self.vocal_synth.synthesize_speech(language_output)
                    if audio_output is not None:
                        self.audio_out.play_audio(audio_output)
                        # Update agency optimizer metrics (echo loop until mic capture integrated)
                        try:
                            # entropy-based uniform adaptation pre-step
                            self.language.entropy_uniform_adapt(scale=1e-4)
                            phoneme_net = self.language.get_phoneme_projection_network()
                            if hasattr(self, 'agency_optimizer') and phoneme_net is not None:
                                # Intended output = language_output (synth params), vocal_output = audio_output
                                self.agency_optimizer.update_metrics(
                                    vocal_output=audio_output,
                                    audio_feedback=None,  # internal echo fallback
                                    intended_output=language_output,
                                    reward_signal=None
                                )
                                self.agency_optimizer.step()
                        except Exception as opt_e:
                            logger.warning(f"Agency optimizer update failed: {opt_e}")
                except Exception as synth_error:
                    logger.error(f"❌ Error in vocal synthesis: {synth_error}")

        except Exception as e:
            logger.error(f"❌ Error making decisions: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")

    def _attempt_dimension_growth(self):
        """Attempt to grow cognitive dimensions based on system performance"""
        # Declare global variables we'll modify
        global current_unified_cog_state_dim, current_sfe_dim, current_emotion_dim
        
        try:
            # Only attempt growth every 100 cycles to avoid overhead
            if self.cycle_count % 100 != 0:
                return

            # Attempt growth through mind module
            if self.mind.attempt_growth():
                # Update global dimension tracking
                old_dim = current_unified_cog_state_dim
                new_dims = self.mind.get_current_dimensions()
                current_unified_cog_state_dim = new_dims['base_dim']

                # Scale sensory and emotional dimensions based on cognitive growth
                stage = new_dims['stage']
                if stage < len(SENSORY_STAGES):
                    old_sfe = current_sfe_dim
                    old_emotion = current_emotion_dim
                    current_sfe_dim = SENSORY_STAGES[stage]
                    current_emotion_dim = EMOTIONAL_STAGES[stage]

                    logger.info(f"🚀 Sensory-Emotional dimensions grown:")
                    logger.info(f"   SFE: {old_sfe} -> {current_sfe_dim}")
                    logger.info(f"   Emotion: {old_emotion} -> {current_emotion_dim}")

                logger.info(f"🚀 Cognitive dimensions grown: {old_dim} -> {current_unified_cog_state_dim}")
                logger.info(f"   Stage: {new_dims['stage']}, Mid: {new_dims['mid_dim']}, High: {new_dims['high_dim']}")

                # Note: Individual modules will need to be updated if they don't support dynamic dimensions
                # This is a complex operation that may require system restart for full compatibility

        except Exception as e:
            logger.error(f"❌ Error attempting dimension growth: {e}")

    def _publish_ui_snapshot(self, cognitive_state, emotional_state, sensory_data, terminal_buffer=None):
        """Assemble and push a lightweight snapshot dict onto the UI data queue."""
        if not self.ui_data_queue:
            return  # UI not active

        # Compute basic norms
        emotional_state_norm = float(np.linalg.norm(emotional_state)) if emotional_state is not None else 0.0
        cognitive_load_norm = float(np.linalg.norm(cognitive_state)) if cognitive_state is not None else 0.0

        # Audio loudness (normalized simple energy)
        audio_vec = sensory_data.get('audio') if sensory_data else None
        if audio_vec is not None and len(audio_vec) > 0:
            sfe_audio_loudness = float(min(1.0, np.linalg.norm(audio_vec) / (len(audio_vec) ** 0.5 + 1e-6)))
        else:
            sfe_audio_loudness = 0.0

        # Video motion (difference from last frame)
        video_vec = sensory_data.get('video') if sensory_data else None
        if video_vec is not None and len(video_vec) > 0:
            if self._last_video_features is not None and len(self._last_video_features) == len(video_vec):
                diff_norm = np.linalg.norm(video_vec - self._last_video_features)
                sfe_video_motion = float(min(1.0, diff_norm / (len(video_vec) ** 0.5 + 1e-6)))
            else:
                sfe_video_motion = 0.0
            # Update last frame copy
            try:
                self._last_video_features = video_vec.copy()
            except Exception:
                self._last_video_features = np.array(video_vec)
        else:
            sfe_video_motion = 0.0

        # Health / tension / tiredness
        current_health_score = float(getattr(self.health, 'get_current_health_score', lambda: 0.0)())
        tiredness_factor = float(getattr(self.health, 'get_tiredness_factor', lambda: 0.0)())
        moral_tension_level = float(getattr(self.conscience, 'get_moral_tension_level', lambda: 0.0)())

        # Initialize metrics
        predict_error_norm = 0.0
        novelty_score = 0.0
        manifold_deviation = 0.0
        goals_satisfaction_overall = 0.0
        output_confidence = 0.0

        is_dreaming = getattr(self.dream, 'current_dream_state', None) not in (None, getattr(self.dream, 'DREAM_STATE_NONE', None))

        # Dream replay / consolidation stats (if exposed)
        dream_replay_stats = None
        if hasattr(self.dream, 'experience_buffer'):
            try:
                buf_len = len(getattr(self.dream, 'experience_buffer', []))
                dream_replay_stats = {
                    'buffer_len': buf_len,
                    'consolidated_this_session': bool(getattr(self.dream, '_consolidated_this_session', False)),
                    'current_dream_state': getattr(self.dream, 'current_dream_state', -1)
                }
            except Exception:
                dream_replay_stats = None

        # Mind summary / goals info
        try:
            mind_dims = self.mind.get_current_dimensions()
        except Exception:
            mind_dims = {'base_dim': current_unified_cog_state_dim}

        goals_info = {}
        if hasattr(self.goals, 'get_status'):
            try:
                goals_info = self.goals.get_status()
            except Exception:
                goals_info = {}

        # Mic level for pre-launch preview (reuse audio loudness metric)
        mic_level = sfe_audio_loudness

        # Video preview small JPEG (only every ~10 cycles to limit overhead)
        video_preview_b64 = None
        if (self.cycle_count % 10 == 0) and hasattr(self.video_in, 'get_latest_frame'):
            try:
                frame = self.video_in.get_latest_frame()
                if frame is not None:
                    import cv2, base64
                    # Resize tiny
                    thumb = cv2.resize(frame, (160,120))
                    # Encode JPEG
                    ok, jpg = cv2.imencode('.jpg', thumb)
                    if ok:
                        video_preview_b64 = base64.b64encode(jpg.tobytes()).decode('utf-8')
            except Exception:
                video_preview_b64 = None

        # Optimizer metrics if available
        optimizer_metrics = None
        if hasattr(self, 'agency_optimizer') and hasattr(self.agency_optimizer, 'get_agency_metrics'):
            try:
                optimizer_metrics = self.agency_optimizer.get_agency_metrics()
            except Exception:
                optimizer_metrics = None

        # Terminal output consolidated
        terminal_output = None
        if terminal_buffer and len(terminal_buffer) > 0 and (self.cycle_count % 5 == 0):
            terminal_output = "\n".join(list(terminal_buffer)[-5:])

        # Include available devices periodically so UI can populate combos
        available_devices = None
        if self.cycle_count < 50 or (self.cycle_count % 200 == 0):
            try:
                available_devices = {
                    'video': list_video_devices(),
                    'audio': list_audio_devices()
                }
            except Exception:
                available_devices = None

        # Memory backup size (rough heuristic) if memory system exposes an internal buffer
        memory_backup_size_bytes = 0
        try:
            if hasattr(self.memory, 'memory_buffer'):
                # If list/array like
                mb = getattr(self.memory, 'memory_buffer')
                if hasattr(mb, '__len__'):
                    memory_backup_size_bytes = len(mb) * 8  # assume float64 default
        except Exception:
            memory_backup_size_bytes = 0

        snapshot = {
            'timestamp': time.time(),
            'data': {
                'current_health_score': current_health_score,
                'tiredness_factor': tiredness_factor,
                'moral_tension_level': moral_tension_level,
                'emotional_state_norm': emotional_state_norm,
                'cognitive_load_norm': cognitive_load_norm,
                # Prediction metrics intentionally de-emphasized in UI (module active internally)
                'predict_error_norm': predict_error_norm,
                'novelty_score': novelty_score,
                'manifold_deviation': manifold_deviation,
                'goals_satisfaction_overall': goals_satisfaction_overall,
                'output_confidence': output_confidence,
                'sfe_audio_loudness': sfe_audio_loudness,
                'sfe_video_motion': sfe_video_motion,
                'mic_level': mic_level,
                'video_preview_jpg_b64': video_preview_b64,
                'optimizer_metrics': optimizer_metrics,
                'terminal_output': terminal_output,
                'available_devices': available_devices,
                'memory_backup_size_bytes': memory_backup_size_bytes,
                'is_dreaming': is_dreaming,
                'dream_replay': dream_replay_stats,
                'heartbeat': self.cycle_count,
                'detailed_internal_data': {
                    'mind_summary': {
                        'cycle': self.cycle_count,
                        'dimensions': mind_dims
                    },
                    'goals_info': goals_info
                }
            }
        }

        try:
            self.ui_data_queue.put_nowait(snapshot)
        except Exception as q_err:
            # Non-fatal; just skip this frame
            logger.debug(f"UI queue put skipped: {q_err}")

    def run_test_cycle(self, duration_seconds: int = 60):
        """Run a comprehensive test cycle with detailed monitoring"""
        logger.info(f"🧪 Starting test cycle for {duration_seconds} seconds...")
        test_start = time.time()

        # Initialize tracking variables
        total_elapsed = 0.0
        cycle_count = 0
        successful_cycles = 0
        failed_cycles = 0

        # Performance tracking
        cycle_times = []
        error_counts = {}

        try:
            while total_elapsed < duration_seconds and not _EMERGENCY_SHUTDOWN_SIGNAL.value:
                cycle_start = time.perf_counter()

                try:
                    # Generate test input
                    test_input = np.random.randn(SFE_DIM)  # Aligned with unified SFE_DIM=64

                    # Process through core pipeline
                    som_output = self.som.process_input(test_input)
                    cognitive_state = self.mind.process_cognitive_state(som_output, {'test': True})

                    # Additional processing
                    emotional_state = self.emotion.process_emotions(cognitive_state)
                    attention_focus = self.attention.compute_attention(cognitive_state, emotional_state)

                    successful_cycles += 1

                except Exception as e:
                    failed_cycles += 1
                    error_type = type(e).__name__
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
                    logger.debug(f"Test cycle error: {e}")

                # Update timing
                cycle_time = time.perf_counter() - cycle_start
                cycle_times.append(cycle_time)
                total_elapsed = time.time() - test_start
                cycle_count += 1

                # Progress reporting
                if cycle_count % 100 == 0:
                    progress = (total_elapsed / duration_seconds) * 100
                    avg_cycle_time = sum(cycle_times[-100:]) / len(cycle_times[-100:])
                    logger.info(f"🧪 Test Progress: {progress:.1f}% | Cycle {cycle_count} | Avg: {avg_cycle_time:.4f}s")

                    # Resource monitoring
                    self._monitor_resources()

                # Prevent excessive CPU usage
                if cycle_time < 0.005:
                    time.sleep(0.005)

        except KeyboardInterrupt:
            logger.info("🧪 Test cycle interrupted by user")
        except Exception as e:
            logger.error(f"🧪 Critical error in test cycle: {e}")

        # Comprehensive test results
        test_duration = time.time() - test_start
        logger.info(f"\n" + "="*60)
        logger.info(f"🧪 TEST RESULTS SUMMARY")
        logger.info(f"="*60)
        logger.info(f"⏱️  Test Duration: {test_duration:.2f} seconds")
        logger.info(f"🔢 Total Cycles: {cycle_count}")
        logger.info(f"✅ Successful Cycles: {successful_cycles}")
        logger.info(f"❌ Failed Cycles: {failed_cycles}")

        if cycle_times:
            avg_cycle_time = sum(cycle_times) / len(cycle_times)
            min_cycle_time = min(cycle_times)
            max_cycle_time = max(cycle_times)
            logger.info(f"📊 Performance:")
            logger.info(f"   Average cycle time: {avg_cycle_time:.4f}s")
            logger.info(f"   Min cycle time: {min_cycle_time:.4f}s")
            logger.info(f"   Max cycle time: {max_cycle_time:.4f}s")
            logger.info(f"   Cycles per second: {1.0/avg_cycle_time:.1f}")

        if error_counts:
            logger.info(f"🚨 Error Summary:")
            for error_type, count in sorted(error_counts.items()):
                logger.info(f"   {error_type}: {count} occurrences")

        # Resource usage summary
        if self.resource_monitor['cpu_percent']:
            avg_cpu = sum(self.resource_monitor['cpu_percent']) / len(self.resource_monitor['cpu_percent'])
            avg_mem = sum(self.resource_monitor['memory_percent']) / len(self.resource_monitor['memory_percent'])
            logger.info(f"💻 Average Resource Usage:")
            logger.info(f"   CPU: {avg_cpu:.1f}%")
            logger.info(f"   Memory: {avg_mem:.1%}")

        success_rate = (successful_cycles / cycle_count * 100) if cycle_count > 0 else 0
        logger.info(f"🎯 Success Rate: {success_rate:.1f}%")

        if success_rate > 95:
            logger.info(f"✅ Test completed successfully!")
        elif success_rate > 80:
            logger.info(f"⚠️  Test completed with minor issues")
        else:
            logger.info(f"❌ Test completed with significant issues")

        logger.info(f"="*60)

    def start(self):
        """Starts Lillith's consciousness and supporting systems with enhanced monitoring"""
        logger.info("🚀 Lillith initiating...")

        try:
            # Validate dimensions before starting
            validate_dimensions()

            # Load all previous states
            # self._load_all_states()

            # Ensure cleanup on exit
            atexit.register(self.stop)

            # Start resource monitoring thread
            monitoring_thread = threading.Thread(target=self._resource_monitoring_thread, daemon=True)
            monitoring_thread.start()

            # Run main loop
            self.run_consciousness_loop()

        except Exception as e:
            logger.critical(f"💥 Failed to start Lillith: {e}")
            logger.critical(f"Full traceback: {traceback.format_exc()}")
            raise

    def _resource_monitoring_thread(self):
        """Background thread for continuous resource monitoring"""
        while not _EMERGENCY_SHUTDOWN_SIGNAL.value:
            try:
                self._monitor_resources()
                time.sleep(RESOURCE_MONITOR_INTERVAL_SEC)
            except Exception as e:
                logger.debug(f"Resource monitoring thread issue: {e}")
                logger.debug(f"Resource monitoring thread issue: {e}")
                time.sleep(RESOURCE_MONITOR_INTERVAL_SEC * 2)

    def stop(self):
        """Performs graceful shutdown with comprehensive cleanup"""
        if not _EMERGENCY_SHUTDOWN_SIGNAL.value:  # Don't try to save/stop if hard killed
            logger.info("🛑 Shutting down Lillith's systems gracefully...")
            shutdown_start = time.time()

            try:
                # 1. Save all states first
                # self._save_all_states()

                # 2. Stop I/O streams with timeout
                io_shutdowns = [
                    ("Audio Input", self.audio_in, "stop"),
                    ("Video Input", self.video_in, "stop"),
                    ("Audio Output", self.audio_out, "stop")
                ]

                for name, component, method_name in io_shutdowns:
                    try:
                        if hasattr(component, method_name):
                            getattr(component, method_name)()
                            logger.debug(f"✅ Stopped {name}")
                    except Exception as e:
                        logger.error(f"❌ Error stopping {name}: {e}")

                # 3. Terminate UI process
                if self.ui_process and self.ui_process.is_alive():
                    logger.info("Terminating UI process...")
                    self.ui_process.terminate()
                    self.ui_process.join(timeout=5.0)
                    if self.ui_process.is_alive():
                        logger.warning("UI process did not terminate gracefully")
                    else:
                        logger.debug("✅ UI process terminated")

                # 4. Allow settle time before final data flush
                settle_seconds = 10
                logger.info(f"⏳ Shutdown settle window {settle_seconds}s before final data flush...")
                end_settle = time.time() + settle_seconds
                while time.time() < end_settle:
                    time.sleep(0.5)

                # 5. Final data flush LAST
                if hasattr(self.data_collector, 'flush_buffer'):
                    try:
                        self.data_collector.flush_buffer()
                        logger.info("🧾 Final data buffer flushed at end of shutdown.")
                    except Exception as fe:
                        logger.error(f"❌ Final data flush failed: {fe}")

                # Final performance summary
                total_runtime = time.time() - self.start_time
                logger.info(f"📊 Session Summary:")
                logger.info(f"   Total runtime: {total_runtime:.2f} seconds")
                logger.info(f"   Total cycles: {self.cycle_count}")
                logger.info(f"   Average cycle rate: {self.cycle_count/total_runtime:.1f} cycles/second")

                if self.error_history:
                    logger.info(f"   Errors encountered: {len(self.error_history)}")

                shutdown_time = time.time() - shutdown_start
                logger.info(f"✅ Lillith's systems gracefully shut down in {shutdown_time:.2f}s (flush performed last)")

            except Exception as e:
                logger.error(f"❌ Error during shutdown: {e}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
        else:
            logger.critical("🚨 Lillith's systems terminated via EMERGENCY SHUTDOWN. State might not be saved.")


# Global emergency shutdown signal
_EMERGENCY_SHUTDOWN_SIGNAL = mp.Value('b', False)


if __name__ == "__main__":
    mp.freeze_support()
    try:
        lillith_orchestrator = LillithOrchestrator()
        lillith_orchestrator.start()
    except KeyboardInterrupt:
        logger.info("👋 Lillith shutdown requested by user")
    except Exception as e:
        logger.critical(f"💥 Critical error starting Lillith: {e}")
        logger.critical(f"Full traceback: {traceback.format_exc()}")
        _EMERGENCY_SHUTDOWN_SIGNAL.value = True
