# run.py
# Launches display first, then display launches model/data/health. Single save state, unified error logging.

import os
import sys
# Prevent Python from writing .pyc files to __pycache__ when this module is executed
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True

import logging
import pickle
from queue import Queue
import time
import threading
sys.dont_write_bytecode = True

# Centralized error logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run")

# State persistence (disabled per user request)
def save_state(state):
    # Disabled: persistence removed to force fresh run each time
    return None

def load_state():
    # Disabled: do not attempt to load saved model state
    return None


def launch_model(data_q, cmd_q):
    try:
        # startup message (avoid printing 'initializ')
        logger.info("MODEL STARTING: bringing subsystems online...")

        # All model imports moved inside function, after display launch
        from nn import mse_loss_prime, AdamW, Sequential, Linear, Sigmoid, ReLU, Tanh
        from OptiJustinJ import JustinJOptimizer
        from cafve import SensoryFeatureExtractor
        from cafve import ConsciousnessAwareFeatureVectorEncoder
        from som import SelfOrganizingMap
        from emotion import EmotionCore
        from memory import MemorySystem
        from mind import Mind
        from itsagirl import ItsAGirl
        from goals import Goals
        from conscience import Conscience
        from tom import ToM
        from health import Health
    # Dream manager deferred: do not import Dream here to avoid starting background consolidation
        from language import Language
        from attention import Attention
        from output import Output
        from vocalsynth import VocalSynth
    # TemporalFabric removed from startup to simplify cognition pipeline
        from inout import AudioIn, AudioOut, VideoIn
        import time

        # Initialize all subsystems
        nn_core = Sequential(Linear(128, 256), ReLU(), Linear(256, 128))
        optimizer = AdamW
        justin_optimizer = JustinJOptimizer([])
        sfe = SensoryFeatureExtractor()
        cafve = ConsciousnessAwareFeatureVectorEncoder()

        # Dynamic SOM input dimension
        som_input_dim = 256  # Minimum module/process unit
        som = SelfOrganizingMap(input_dim=som_input_dim)

        emotion = EmotionCore(512)
        memory = MemorySystem()
        # Mind requires 7 positional arguments: initial_dim_stage, som_activation_dim, som_bmu_coords_dim, emotional_state_dim, memory_recall_dim, predictive_error_dim, unified_cognitive_state_dim
        # Example values based on typical model dimensions
        initial_dim_stage = 0
        som_activation_dim = 289  # 17x17 SOM
        som_bmu_coords_dim = 2
        emotional_state_dim = 108
        memory_recall_dim = 512
        predictive_error_dim = 80
        unified_cognitive_state_dim = 256
        mind = Mind(
            initial_dim_stage,
            som_activation_dim,
            som_bmu_coords_dim,
            emotional_state_dim,
            memory_recall_dim,
            predictive_error_dim,
            unified_cognitive_state_dim,
            som_instance=som,
            emotion_instance=emotion,
            memory_instance=memory
        )
        itsagirl = ItsAGirl()
        goals = Goals()
        conscience = Conscience()
        tom = ToM()
        health = Health()
        # Deferred dream manager - will be created on demand by the orchestrator or other component
        dream = None
        language = Language()
        attention = Attention()
        output = Output(512, 512)
        vocalsynth = VocalSynth()
    # TemporalFabric disabled: not created during startup
        audioin = AudioIn()
        audioout = AudioOut()
        videoin = VideoIn()

        # Optionally restore state
        state = load_state()
        if state:
            # Add logic to restore subsystems from state if needed
            pass

        # summary - do not use 'initialized' wording
        logger.info("All model subsystems are now online. Dream manager creation deferred until requested.")

        # Dynamic scaling: update SOM input_dim if Mind grows
        def check_and_update_som():
            new_dim = mind.base_dim
            if new_dim != som.input_dim:
                # Recreate SOM with the new dimension
                return SelfOrganizingMap(input_dim=new_dim)

        cycle = 0
        while True:
            # Collect metrics from all subsystems
            snapshot = {
                "timestamp": time.time(),
                "nn": str(type(nn_core)),
                "optimizer": str(type(optimizer)),
                "justin_optimizer": str(type(justin_optimizer)),
                "sfe": str(type(sfe)),
                "cafve": str(type(cafve)),
                "som": str(type(som)),
                "emotion": str(type(emotion)),
                "memory": str(type(memory)),
                "mind": str(type(mind)),
                "itsagirl": str(type(itsagirl)),
                "goals": str(type(goals)),
                "conscience": str(type(conscience)),
                "tom": str(type(tom)),
                "health": str(type(health)),
                "dream": str(type(dream)),
                "language": str(type(language)),
                "attention": str(type(attention)),
                "output": str(type(output)),
                "vocalsynth": str(type(vocalsynth)),
                # temporal removed from snapshot
                "audioin": str(type(audioin)),
                "audioout": str(type(audioout)),
                "videoin": str(type(videoin)),
                "cycle": cycle,
            }
            try:
                data_q.put_nowait(snapshot)
            except Exception as e:
                logger.error(f"Failed to send snapshot to display: {e}")
            # Handle commands from display
            try:
                cmd = cmd_q.get_nowait()
                # Implement command handling here
            except Exception:
                pass
            cycle += 1
            time.sleep(0.1)
    except Exception as e:
        # Ensure that exceptions in the model thread are logged but do not call sys.exit()
        logger.error(f"Model/data/health startup failed: {e}")
        return
    


def main():
    # Run the PyQt UI in the main thread (required by Qt on many platforms)
    # and handle model startup in a background worker thread when UI sends LAUNCH_MODEL.
    from display import start_qt_app
    data_q = Queue()
    cmd_q = Queue()

    # Start a thread to listen for UI commands so we don't block the Qt event loop
    def command_listener():
            # Wait for explicit LAUNCH_MODEL command from UI or other controller.
            logger.info("Command listener started: awaiting explicit LAUNCH_MODEL or DISPLAY_CLOSED.")
            while True:
                try:
                    cmd = cmd_q.get()
                    if cmd == "LAUNCH_MODEL":
                        logger.info("LAUNCH_MODEL received. Starting model in background thread.")
                        try:
                            t = threading.Thread(target=launch_model, args=(data_q, cmd_q), daemon=True)
                            t.start()
                            logger.info("Model background thread started (explicit launch).")
                        except Exception as e:
                            logger.error(f"Failed to start model thread: {e}")
                        break
                    if cmd == "DISPLAY_CLOSED":
                        logger.error("Display reported closed. Aborting startup.")
                        return
                except Exception as e:
                    logger.error(f"Error in command listener: {e}")
                    return

            # Continue processing commands (shutdown, additional controls)
            logger.info("Command listener running: processing DISPLAY_CLOSED and other commands.")
            while True:
                try:
                    cmd = cmd_q.get()
                    if cmd == "DISPLAY_CLOSED":
                        logger.error("Display reported closed. Aborting remaining startup tasks.")
                        return
                    else:
                        # Other commands can be processed by model loop or forwarded
                        logger.debug(f"Command listener received: {cmd}")
                except Exception as e:
                    logger.error(f"Error in command listener: {e}")
                    return

    # Start the command listener thread before launching the UI
    listener_thread = threading.Thread(target=command_listener, daemon=True)
    listener_thread.start()

    # Now run the Qt app in the main thread. start_qt_app will block until UI closes.
    try:
        logger.info("Starting display UI in main thread; command listener running in background.")
        start_qt_app(data_q, cmd_q)
    except Exception as e:
        logger.error(f"Display or model launch failed: {e}")

if __name__ == "__main__":
    main()
