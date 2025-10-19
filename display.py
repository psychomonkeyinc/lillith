# --- START OF FILE display.py ---
# display.py

from ptqt5 import QtWidgets, QtCore, QtGui

def list_video_devices():
    """List all available video capture devices with enhanced error handling"""
    import cv2
    available_cameras = []
    try:
        for i in range(10):
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
                cap.release()
    except Exception as e:
        print(f"Error checking camera {i}: {e}")
    return available_cameras

def list_audio_devices():
    """List all available audio input devices with enhanced error handling"""
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
        return input_devices
    except ImportError:
        print("sounddevice not available for audio device detection")
        return []
    except Exception as e:
        print(f"Error scanning audio devices: {e}")
        return []

import os
import json
import numpy as np
import logging
import threading
import queue
import sys
import time
from typing import Dict, List, Any, Optional
import tkinter as tk
from tkinter import ttk
from PyQt5.QtWidgets import QPlainTextEdit

# Assume PyQt5 and pyqtgraph are installed
try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    import pyqtgraph as pg
except ImportError:
    logging.Logger.warning("PyQt5 or pyqtgraph not found. Display module will be non-functional.")
    # Define dummy classes to allow the file to be parsed without crash
    class QtWidgets:
        class QApplication:
            def __init__(self, *args): pass
            def instance(self): return self
            def exec_(self): pass
        class QWidget:
            def __init__(self, *args): pass
            def setWindowTitle(self, *args): pass
            def setGeometry(self, *args): pass
            def setStyleSheet(self, *args): pass
            def setLayout(self, *args): pass
            def show(self): pass
        class QPushButton:
            def __init__(self, *args): pass
            def clicked(self): return self
            def connect(self, *args): pass
        class QLineEdit:
            def __init__(self, *args): pass
            def text(self): return ""
            def setValidator(self, *args): pass
        class QLabel:
            def __init__(self, *args): pass
            def setText(self, *args): pass
        class QTextEdit:
            def __init__(self, *args): pass
            def setReadOnly(self, *args): pass
        class QVBoxLayout:
            def __init__(self, *args): pass
            def addWidget(self, *args): pass
        class QHBoxLayout:
            def __init__(self, *args): pass
            def addWidget(self, *args): pass
    class QtCore:
        class QTimer:
            def __init__(self, *args): pass
            def setInterval(self, *args): pass
            def timeout(self): return self
            def connect(self, *args): pass
            def start(self): pass
    class QtGui:
        class QIntValidator:
            def __init__(self, *args): pass
    class pg:
        class GraphicsLayoutWidget:
            def __init__(self, *args): pass
            def addPlot(self, *args, **kwargs): return MockPlot()
        def setConfigOption(self, *args): pass

# Mock classes for pyqtgraph plot elements where not using actual pg
class MockPlot:
    def plot(self, *args, **kwargs): return MockCurve()
    def setXRange(self, *args): pass
    def setYRange(self, *args): pass
    def setTitle(self, *args): pass
    def setLabel(self, *args): pass
    def showGrid(self, *args): pass

class MockCurve:
    def setData(self, *args): pass


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global for QApplication instance ---
QT_APP_INSTANCE: Optional[QtWidgets.QApplication] = None
BUFFER_SIZE = 600 # Store 10 minutes of data at 1Hz, or 1 minute at 10Hz

class _RunStatus:
    def __init__(self):
        self.phase = "INIT"
        self.phase_started = time.time()
        self.phase_elapsed = 0.0
        self.phase_remaining = 0.0
        self.cycle = 0
        self.recent_errors = []
        self.health = 0.0
        self.cog_norm = 0.0
        self.emo_norm = 0.0
        self.is_dreaming = False
        self.last_snapshot_ts = 0.0

_run_status = _RunStatus()
_log_buffer = queue.deque(maxlen=200)

def _fmt_bar(frac: float, width: int = 30) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(frac * width)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {int(frac*100):3d}%"

def _print_status():
    rs = _run_status
    phase = rs.phase
    bar = ""
    if rs.phase in ("AWAKE", "DREAM") and rs.phase_remaining > 0:
        total = rs.phase_elapsed + rs.phase_remaining
        frac = rs.phase_elapsed / total if total > 0 else 0.0
        bar = _fmt_bar(frac)
    sys.stdout.write("\033[2J\033[H")  # clear screen (ANSI) - safe in most terminals
    sys.stdout.write(f"=== LILLITH LIVE ===\n")
    sys.stdout.write(f"Phase: {phase}   Cycle: {rs.cycle}   DreamFlag: {rs.is_dreaming}\n")
    sys.stdout.write(f"Elapsed: {rs.phase_elapsed:6.1f}s  Remaining: {rs.phase_remaining:6.1f}s\n")
    if bar:
        sys.stdout.write(f"{bar}\n")
    sys.stdout.write(f"Health:{rs.health:5.2f}  CogNorm:{rs.cog_norm:5.2f}  EmoNorm:{rs.emo_norm:5.2f}\n")
    if rs.recent_errors:
        sys.stdout.write("Recent Errors:\n")
        for e in rs.recent_errors[-3:]:
            sys.stdout.write(f"  - {e.get('error','?')} (cycle {e.get('cycle','?')})\n")
    sys.stdout.write("\n--- Recent Logs ---\n")
    for line in list(_log_buffer)[-8:]:
        sys.stdout.write(line + "\n")
    sys.stdout.flush()

def _ui_consumer_loop(data_queue):
    while True:
        try:
            msg = data_queue.get()
            if msg is None:
                break
            data = msg.get('data', {})
            # Log events
            log_event = data.get('log_event')
            if log_event:
                level = log_event.get('level', '')
                text = log_event.get('message', '')
                _log_buffer.append(f"{level[:1]} {text}")
            # Phase events
            phase_event = data.get('phase_event')
            if phase_event:
                _run_status.phase = phase_event.get('phase', _run_status.phase)
                _run_status.phase_started = time.time()
                _log_buffer.append(f"* Phase → {_run_status.phase}: {phase_event.get('message','')}")
            # Snapshots
            run_phase = data.get('run_phase')
            if run_phase:
                _run_status.phase = run_phase
                _run_status.phase_elapsed = data.get('phase_elapsed_sec', 0.0)
                _run_status.phase_remaining = data.get('phase_remaining_sec', 0.0)
                _run_status.cycle = data.get('cycle', _run_status.cycle)
                _run_status.health = data.get('current_health_score', _run_status.health)
                _run_status.cog_norm = data.get('cognitive_load_norm', _run_status.cog_norm)
                _run_status.emo_norm = data.get('emotional_state_norm', _run_status.emo_norm)
                _run_status.is_dreaming = data.get('is_dreaming', _run_status.is_dreaming)
                rec_err = data.get('recent_errors')
                if isinstance(rec_err, list):
                    _run_status.recent_errors = rec_err
            # Throttled redraw
            now = time.time()
            if now - _run_status.last_snapshot_ts > 0.5:
                _run_status.last_snapshot_ts = now
                _print_status()
        except Exception as e:
            _log_buffer.append(f"! UI loop error: {e}")

class TerminalWidget(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setStyleSheet("background-color: black; color: white; font-family: monospace;")

    def write(self, text):
        self.appendPlainText(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def flush(self):
        pass  # Required for compatibility with sys.stdout

class LiveDisplay(QtWidgets.QWidget):
    """
    Lillith's Live Display / UI. Provides real-time visualization of internal states,
    health, and allows for manual control commands (like triggering dreams/shutdown).
    """
    def __init__(self, data_queue: queue.Queue, command_queue: queue.Queue):
        # Using QtWidgets.QWidget directly from the imported PyQt5, or mock if not available.
        super().__init__()
        self.data_queue = data_queue      # Queue for receiving data snapshots from main loop
        self.command_queue = command_queue # Queue for sending commands to main loop

        self.setWindowTitle('Lillith Mission Control'); 
        self.setGeometry(50, 50, 1800, 1000); 
        self.setStyleSheet("background-color: #111; color: #eee;")

        # Data buffers for plotting (circular buffers)
        # These will hold scalar metrics over time
        self.timestamps = [0.0] * BUFFER_SIZE
        self.health_score = [0.0] * BUFFER_SIZE
        self.tiredness_factor = [0.0] * BUFFER_SIZE
        self.moral_tension = [0.0] * BUFFER_SIZE
        self.emotion_intensity_norm = [0.0] * BUFFER_SIZE
        self.cognitive_load_norm = [0.0] * BUFFER_SIZE
        self.predict_error_norm = [0.0] * BUFFER_SIZE
        self.novelty_score = [0.0] * BUFFER_SIZE
        self.manifold_deviation = [0.0] * BUFFER_SIZE
        self.goals_satisfaction_overall = [0.0] * BUFFER_SIZE
        self.output_confidence = [0.0] * BUFFER_SIZE

        # Raw audio/video metrics (e.g., max amplitude) from SFE
        self.sfe_audio_loudness = [0.0] * BUFFER_SIZE
        self.sfe_video_motion = [0.0] * BUFFER_SIZE

        # Index to manage circular buffers
        self.data_idx = 0
        self.last_ts = time.perf_counter() # For relative timestamps on plot

        self.setup_ui()

        # Timer to update plots periodically
        self.timer = QtCore.QTimer(); 
        self.timer.setInterval(100); # Update every 100 ms (10 Hz)
        self.timer.timeout.connect(self.update_plots); 
        self.timer.start()

        logger.info("LiveDisplay UI initialized.")

    def closeEvent(self, event):
        """Handle window close - notify main loop via command queue."""
        # Write debug info to disk when closeEvent occurs (helps detect unexpected closes)
        try:
            dbgfile = os.path.join(os.path.dirname(__file__), 'updated', 'display_close_debug.txt')
            with open(dbgfile, 'a', encoding='utf-8') as f:
                f.write(f"closeEvent invoked at {time.time()}\n")
        except Exception:
            pass
        try:
            if self.command_queue is not None:
                self.command_queue.put_nowait("DISPLAY_CLOSED")
                logger.info("Display: sent DISPLAY_CLOSED to command queue.")
        except Exception:
            logger.debug("Display: failed to send DISPLAY_CLOSED.")
        # Proceed with default close behavior
        try:
            super().closeEvent(event)
        except Exception:
            pass

    def setup_ui(self):
        """Sets up the graphical user interface layout and widgets."""
        main_layout = QtWidgets.QHBoxLayout(); self.setLayout(main_layout)

        # Control Panel (Left Side)
        control_panel = self._create_control_panel(); main_layout.addWidget(control_panel, 1)

        # Camera/Mic Panel (Top Center)
        cam_mic_panel = QtWidgets.QVBoxLayout()
        cam_mic_widget = QtWidgets.QWidget()
        cam_mic_widget.setLayout(cam_mic_panel)

        # Camera feed + embedded device selectors
        self.video_devices = list_video_devices()
        self.audio_devices = list_audio_devices()

        # Outer frame for camera area
        cam_frame = QtWidgets.QWidget()
        cam_frame.setStyleSheet("background-color: #111; border: 1px solid #444;")
        cam_frame_layout = QtWidgets.QVBoxLayout()
        cam_frame.setLayout(cam_frame_layout)

        # Camera feed (larger window) - central visual
        self.camera_label = QtWidgets.QLabel("Camera feed loading...")
        self.camera_label.setMinimumSize(640, 360)
        self.camera_label.setMaximumHeight(420)
        self.camera_label.setStyleSheet("background-color: #000; border: 1px solid #333;")
        cam_frame_layout.addWidget(self.camera_label, 6)

        # Overlay area below the camera to hold device dropdowns (visually part of camera pane)
        device_bar = QtWidgets.QWidget()
        device_bar_layout = QtWidgets.QHBoxLayout()
        device_bar.setLayout(device_bar_layout)
        device_bar.setStyleSheet("background-color: rgba(20,20,20,0.9); border-top: 1px solid #333;")

        # Video dropdown
        self.video_device_dropdown = QtWidgets.QComboBox()
        for cam in self.video_devices:
            self.video_device_dropdown.addItem(f"Camera {cam['index']} ({cam['backend']}) - {cam['resolution']} @ {cam['fps']}fps", cam['index'])
        # Default to the first available device (restore original behavior)
        if self.video_device_dropdown.count() > 0:
            self.video_device_dropdown.setCurrentIndex(0)
        device_bar_layout.addWidget(QtWidgets.QLabel("Video:"))
        device_bar_layout.addWidget(self.video_device_dropdown)
        # Immediate switch when user changes selection
        try:
            self.video_device_dropdown.currentIndexChanged.connect(lambda idx: self._on_video_device_changed())
        except Exception:
            pass

        # Audio dropdown
        self.audio_device_dropdown = QtWidgets.QComboBox()
        for dev in self.audio_devices:
            self.audio_device_dropdown.addItem(f"{dev['name']} - {dev['channels']}ch @ {dev['default_samplerate']}Hz", dev['index'])
        if self.audio_device_dropdown.count() > 0:
            self.audio_device_dropdown.setCurrentIndex(0)
        device_bar_layout.addWidget(QtWidgets.QLabel("Audio:"))
        device_bar_layout.addWidget(self.audio_device_dropdown)

        # Apply button
        self.apply_devices_button = QtWidgets.QPushButton("Apply Devices")
        self.apply_devices_button.clicked.connect(self._apply_selected_devices)
        device_bar_layout.addWidget(self.apply_devices_button)

        cam_frame_layout.addWidget(device_bar, 1)

        # Mic EQ bar (thin) below but very close to camera feed
        self.mic_bar = QtWidgets.QProgressBar()
        self.mic_bar.setFixedHeight(16)
        self.mic_bar.setRange(0, 100)
        self.mic_bar.setTextVisible(False)
        self.mic_bar.setStyleSheet("background-color: #222; border: 1px solid #444;")
        cam_frame_layout.addWidget(self.mic_bar, 0)

        cam_mic_panel.addWidget(cam_frame)

        main_layout.addWidget(cam_mic_widget, 2)

        # Start camera and mic update timers immediately using currently selected device (improves UX)
        try:
            video_idx = self.video_device_dropdown.currentData()
            audio_idx = self.audio_device_dropdown.currentData()
        except Exception:
            video_idx = None
            audio_idx = None
        self._init_camera_mic(video_index=video_idx, audio_index=audio_idx)

        # Plot Panel (Right Side)
        plot_panel = self._create_plot_panel(); main_layout.addWidget(plot_panel, 4)

        # Adjust stretch so plot panel gets more space than mid camera pane
        try:
            main_layout.setStretch(0, 1)  # control panel
            main_layout.setStretch(1, 2)  # camera pane
            main_layout.setStretch(2, 4)  # plots
        except Exception:
            pass


        def _apply_selected_devices(self):
            video_idx = self.video_device_dropdown.currentData()
            audio_idx = self.audio_device_dropdown.currentData()
            # Send device selection to main loop
            self._send_command(f"APPLY_DEVICES::{video_idx}::{audio_idx}")
        # Terminal Panel (Bottom)
        self.terminal = TerminalWidget()
        main_layout.addWidget(self.terminal, 2)

        # Redirect stdout and stderr to the terminal
        sys.stdout = self.terminal
        sys.stderr = self.terminal

        # Camera/mic update timers and attention integration are started during
        # initialization and when devices are applied. Devices are initialized
        # at startup using the currently selected device entries and may be
        # re-initialized by the user via the 'Apply Devices' control.

    def _init_camera_mic(self, video_index=None, audio_index=None):
        # Release previous camera/mic if present
        if hasattr(self, 'cap') and self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        if hasattr(self, 'stream') and self.stream:
            try:
                self.stream.stop()
            except Exception:
                pass

        # Camera feed using OpenCV
        try:
            import cv2
            idx = video_index if video_index is not None else self.video_device_dropdown.currentData()
            # Use DirectShow for Windows if available for better compatibility
            try:
                self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    # Fallback to default backend
                    self.cap.release()
                    self.cap = cv2.VideoCapture(idx)
            except Exception:
                self.cap = cv2.VideoCapture(idx)
        except Exception as e:
            self.cap = None
            self.camera_label.setText(f"Camera error: {e}")

        # Mic EQ using sounddevice
        try:
            import sounddevice as sd
            import numpy as np
            self.mic_level = 0
            mic_idx = audio_index if audio_index is not None else self.audio_device_dropdown.currentData()
            def audio_callback(indata, frames, time, status):
                self.mic_level = int(np.clip(np.abs(indata).mean() * 200, 0, 100))
            self.stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=44100, device=mic_idx)
            self.stream.start()
        except Exception as e:
            self.mic_bar.setFormat(f"Mic error: {e}")

        # Timer for updating visuals
        self.cam_mic_timer = QtCore.QTimer()
        self.cam_mic_timer.setInterval(50)
        self.cam_mic_timer.timeout.connect(self._update_camera_mic)
        self.cam_mic_timer.start()
    def _apply_selected_devices(self):
        video_idx = self.video_device_dropdown.currentData()
        audio_idx = self.audio_device_dropdown.currentData()
        self._init_camera_mic(video_index=video_idx, audio_index=audio_idx)
        self._send_command(f"APPLY_DEVICES::{video_idx}::{audio_idx}")

    def _on_video_device_changed(self):
        """Called when user changes the video device dropdown - switch camera immediately."""
        try:
            video_idx = self.video_device_dropdown.currentData()
            # Keep current audio device unchanged
            audio_idx = self.audio_device_dropdown.currentData() if hasattr(self, 'audio_device_dropdown') else None
            self._init_camera_mic(video_index=video_idx, audio_index=audio_idx)
        except Exception as e:
            logger.error(f"Error switching camera: {e}")

    def _update_camera_mic(self):
        # Update camera feed
        if hasattr(self, 'cap') and self.cap:
            ret, frame = self.cap.read()
            if ret:
                import cv2
                from PyQt5.QtGui import QImage, QPixmap
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_img).scaled(self.camera_label.width(), self.camera_label.height())
                self.camera_label.setPixmap(pixmap)
            else:
                self.camera_label.setText("No camera frame")

        # Update mic EQ bar
        if hasattr(self, 'mic_level'):
            self.mic_bar.setValue(self.mic_level)

    def _create_control_panel(self) -> QtWidgets.QWidget:
        """Creates the control panel with buttons, labels, and text displays."""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        panel.setLayout(layout)

        # Status Label
        self.status_label = QtWidgets.QLabel("SYSTEM: Initializing...")
        self.status_label.setStyleSheet("font-size: 20px; color: yellow; font-weight: bold;")
        layout.addWidget(self.status_label)
        layout.addWidget(QtWidgets.QLabel("") ) # Spacer

        # Launch button (only launches model)
    # NOTE: Model autostart behavior is controlled by the runtime (run.py).
    # The explicit UI 'Launch' control has been removed to avoid duplicate
    # startup controls. Model start/stop should be handled by the orchestrator.

        # Dream/Sleep Control
        layout.addWidget(QtWidgets.QLabel("--- Rest / Control ---"))
        self.nap_button = QtWidgets.QPushButton("Trigger Nap (Auto-Duration)")
        self.nap_button.clicked.connect(lambda: self._send_command("NAP"))
        layout.addWidget(self.nap_button)
        
        self.sleep_button = QtWidgets.QPushButton("Trigger Sleep (5m Countdown)")
        self.sleep_button.clicked.connect(lambda: self._send_command("SLEEP_COUNTDOWN"))
        layout.addWidget(self.sleep_button)

        self.instant_sleep_button = QtWidgets.QPushButton("Instant Sleep (No Countdown)")
        self.instant_sleep_button.clicked.connect(lambda: self._send_command("INSTANT_SLEEP"))
        layout.addWidget(self.instant_sleep_button)

        self.emergency_shutdown_button = QtWidgets.QPushButton("EMERGENCY SHUTDOWN (HARD KILL)")
        self.emergency_shutdown_button.clicked.connect(lambda: self._send_command("EMERGENCY_SHUTDOWN"))
        self.emergency_shutdown_button.setStyleSheet("background-color: black; color: red; font-weight: bold;")
        layout.addWidget(self.emergency_shutdown_button)

        self.wake_up_button = QtWidgets.QPushButton("FORCE WAKE UP")
        self.wake_up_button.clicked.connect(lambda: self._send_command("WAKE_UP"))
        layout.addWidget(self.wake_up_button)
        layout.addWidget(QtWidgets.QLabel("")) # Spacer

        # Interpreter / Debug Info (Placeholder for more detailed info)
        layout.addWidget(QtWidgets.QLabel("--- Lillith's Internal World ---"))
        self.interpreter_text = QtWidgets.QTextEdit(); self.interpreter_text.setReadOnly(True)
        self.interpreter_text.setStyleSheet("background-color: #222; color: #ccf; border: 1px solid #444;")
        layout.addWidget(self.interpreter_text, 5) # Takes more vertical space

        self.goal_state_text = QtWidgets.QTextEdit(); self.goal_state_text.setReadOnly(True)
        self.goal_state_text.setStyleSheet("background-color: #222; color: #fcc; border: 1px solid #444;")
        layout.addWidget(QtWidgets.QLabel("Current Goals (Satisfaction):"))
        layout.addWidget(self.goal_state_text, 2)

        return panel

    def _create_plot_panel(self) -> pg.GraphicsLayoutWidget:
        """Creates the panel with live plots of Lillith's vitals."""
        # Assume pg.GraphicsLayoutWidget is available or custom mock.
        widget = pg.GraphicsLayoutWidget(); 
        pg.setConfigOption('background', '#111'); 
        pg.setConfigOption('foreground', 'w') # White foreground for text/lines unless specified

        # Define Plots for key metrics
        self.plots = {
            "health": widget.addPlot(row=0, col=0, title="System Health (0-1)"),
            "tiredness": widget.addPlot(row=0, col=1, title="Tiredness Factor (0-1)"),
            "moral_tension": widget.addPlot(row=1, col=0, title="Moral Tension (0-1)"),
            "emotion_norm": widget.addPlot(row=1, col=1, title="Overall Emotion Intensity (Norm)"),
            "cog_load": widget.addPlot(row=2, col=0, title="Cognitive Load (Unified State Norm)"),
            "predict_error": widget.addPlot(row=2, col=1, title="Prediction Error (Norm)"),
            "novelty": widget.addPlot(row=3, col=0, title="Novelty Score (0-1)"),
            "manifold_dev": widget.addPlot(row=3, col=1, title="Manifold Deviation (Identity Tension)"),
            "goals_sat": widget.addPlot(row=4, col=0, title="Overall Goal Satisfaction (0-1)"),
            "output_conf": widget.addPlot(row=4, col=1, title="Output Confidence (0-1)"),
            "audio_loudness": widget.addPlot(row=5, col=0, title="SFE Audio Loudness (0-1)"),
            "video_motion": widget.addPlot(row=5, col=1, title="SFE Video Motion (0-1)")
        }

        # Configure plot appearance and create curves
        self.curves = {}
        plot_colors = ['#00FF00', '#FFD700', '#FF4500', '#00BFFF', '#BA55D3', '#32CD32', '#F08080', '#ADD8E6', '#FF6347', '#DDA0DD', '#F4A460', '#A52A2A'] # Green, Gold, OrangeRed for basic. Other specific colors.

        for i, (key, plot) in enumerate(self.plots.items()):
            curve_color = plot_colors[i % len(plot_colors)]
            self.curves[key] = plot.plot(pen=curve_color)
            plot.setYRange(0, 1.1) # Default range for 0-1 metrics
            plot.setLabel(axis='bottom', text='Time (cycles)')
            plot.showGrid(x=True, y=True)
        
        # Specific ranges for some plots
        self.plots["moral_tension"].setYRange(0, 1.1)
        self.plots["emotion_norm"].setYRange(0, 15) # Norm of 108D vector (max ~sqrt(108) = 10.4)
        self.plots["cog_load"].setYRange(0, 20) # Norm of 256D vector (max ~sqrt(256)=16)
        self.plots["predict_error"].setYRange(0, 10) # Norm of 80D error (max ~sqrt(80)=8.9)
        self.plots["manifold_dev"].setYRange(0, 20) # Norm of 256D vector (max ~sqrt(256)=16)

        return widget

    def _send_command(self, command_str: str):
        """Sends a command string to the main application loop via the command queue."""
        try:
            self.command_queue.put_nowait(command_str)
            logger.info(f"UI: Sent command '{command_str}' to main loop.")
        except queue.Full:
            logger.warning("UI: Command queue is full. Could not send command.")

    def update_plots(self):
        """
        Retrieves data snapshots from the data_queue and updates all plots.
        Called periodically by the QTimer.
        """
        new_data_received = False
        while not self.data_queue.empty():
            try:
                snapshot = self.data_queue.get_nowait()
                if not isinstance(snapshot, dict):
                    continue

                # Support two snapshot shapes:
                # 1) { 'timestamp': ..., 'data': { ... } }
                # 2) { 'timestamp': ..., 'current_health_score': ..., ... } (flat)
                ts = snapshot.get('timestamp', time.time())
                if isinstance(snapshot.get('data'), dict):
                    data = snapshot['data']
                else:
                    # treat snapshot itself as the data container
                    data = snapshot

                # Add new data at current index, then advance index (circular buffer)
                self.timestamps[self.data_idx] = ts
                self.health_score[self.data_idx] = data.get('current_health_score', 0.0)
                self.tiredness_factor[self.data_idx] = data.get('tiredness_factor', 0.0)
                self.moral_tension[self.data_idx] = data.get('moral_tension_level', 0.0)
                self.emotion_intensity_norm[self.data_idx] = data.get('emotional_state_norm', 0.0)
                self.cognitive_load_norm[self.data_idx] = data.get('cognitive_load_norm', 0.0)
                self.predict_error_norm[self.data_idx] = data.get('predict_error_norm', 0.0)
                self.novelty_score[self.data_idx] = data.get('novelty_score', 0.0)
                self.manifold_deviation[self.data_idx] = data.get('manifold_deviation', 0.0)
                self.goals_satisfaction_overall[self.data_idx] = data.get('goals_satisfaction_overall', 0.0)
                self.output_confidence[self.data_idx] = data.get('output_confidence', 0.0)

                # SFE specific data
                self.sfe_audio_loudness[self.data_idx] = data.get('sfe_audio_loudness', 0.0)
                self.sfe_video_motion[self.data_idx] = data.get('sfe_video_motion', 0.0)

                # Update status label and text areas
                is_dreaming_status = "DREAMING" if data.get('is_dreaming', False) else "AWAKE"
                try:
                    self.status_label.setText(f"LILLITH STATUS: {is_dreaming_status}")
                except Exception:
                    # ignore transient UI update errors
                    pass

                # For interpreting Lillith's internal world (Mind/Conscience/ToM/Goals)
                detailed_internal_data = data.get('detailed_internal_data', {}) if isinstance(data, dict) else {}
                try:
                    self.interpreter_text.setText(json.dumps(detailed_internal_data.get('mind_summary', {}), indent=2))
                    goals_info = detailed_internal_data.get('goals_info', {})
                    self.goal_state_text.setText(json.dumps(goals_info, indent=2))
                except Exception:
                    # ignore errors setting large text blobs
                    pass

                self.data_idx = (self.data_idx + 1) % BUFFER_SIZE
                new_data_received = True

            except Exception as e:
                # Log error and include a concise snapshot summary to aid debugging
                try:
                    s_keys = list(snapshot.keys()) if isinstance(snapshot, dict) else str(type(snapshot))
                except Exception:
                    s_keys = '<unreadable snapshot>'
                logger.error(f"Error updating plots from data queue: {e} - snapshot keys: {s_keys}")
                continue

        if new_data_received:
            # Prepare data for plotting (circular buffer is tricky; extract a continuous segment)
            # Find the most recent (BUFFER_SIZE) points in chronological order.
            # Convert continuous timestamps to relative cycle numbers or simple indices
            x_data = np.arange(BUFFER_SIZE) # Simple indices

            # Update curves with new data
            self.curves["health"].setData(x_data, self._get_circular_data(self.health_score))
            self.curves["tiredness"].setData(x_data, self._get_circular_data(self.tiredness_factor))
            self.curves["moral_tension"].setData(x_data, self._get_circular_data(self.moral_tension))
            self.curves["emotion_norm"].setData(x_data, self._get_circular_data(self.emotion_intensity_norm))
            self.curves["cog_load"].setData(x_data, self._get_circular_data(self.cognitive_load_norm))
            self.curves["predict_error"].setData(x_data, self._get_circular_data(self.predict_error_norm))
            self.curves["novelty"].setData(x_data, self._get_circular_data(self.novelty_score))
            self.curves["manifold_dev"].setData(x_data, self._get_circular_data(self.manifold_deviation))
            self.curves["goals_sat"].setData(x_data, self._get_circular_data(self.goals_satisfaction_overall))
            self.curves["output_conf"].setData(x_data, self._get_circular_data(self.output_confidence))
            self.curves["audio_loudness"].setData(x_data, self._get_circular_data(self.sfe_audio_loudness))
            self.curves["video_motion"].setData(x_data, self._get_circular_data(self.sfe_video_motion))

            # Auto-range X-axis or base it on time elapsed
            # self.plots['health'].setXRange(self.timestamps[0], self.timestamps[-1]) # Use time if preferred


    def _get_circular_data(self, data_list: list) -> np.ndarray:
        """Helper to correctly retrieve data from a circular buffer for plotting."""
        # This correctly orders data if the buffer wrapped around
        return np.array(data_list[self.data_idx:] + data_list[:self.data_idx])

# --- END REVERTED SECTIONS (removed recent UI additions) ---

def start_qt_app(data_q, cmd_q):
    import sys
    from PyQt5 import QtWidgets
    if QtWidgets.QApplication.instance() is None:
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()
    window = LiveDisplay(data_q, cmd_q)
    window.show()
    # Notify the caller that the UI is up and ready to accept commands/snapshots.
    # NOTE: Do NOT auto-send UI_READY here. The model must be launched explicitly
    # by sending a `LAUNCH_MODEL` command on `cmd_q` (e.g., from a user-initiated button).
    app.exec_()

def run_ui_thread(data_q: queue.Queue, cmd_q: queue.Queue) -> threading.Thread:
    """
    Creates and starts a thread that runs the PyQt5 UI.
    This is the function that main.py should call.
    """
    ui_thread = threading.Thread(target=start_qt_app, args=(data_q, cmd_q), daemon=True)
    ui_thread.start()
    logger.info("PyQt5 UI thread started.")
    return ui_thread

def start_display_process(ui_data_queue, ui_command_queue):
    """
    Starts the display UI in a separate process.
    This is the function imported in main.py.
    
    Args:
        data_queue: Queue for sending data from main process to UI
        command_queue: Queue for receiving commands from UI to main process
        
    Returns:
        Process object for the UI process
    """
    import multiprocessing as mp
    
    # Create a new process for the UI
    display_process = mp.Process(
        target=start_qt_app,
        args=(ui_data_queue, ui_command_queue),
        daemon=True,
        name="LillithUI"
    )
    
    # Start the process
    display_process.start()

    # Add lightweight textual monitor in SAME process (not a new process) if you want immediate visibility.
    t = threading.Thread(target=_ui_consumer_loop, args=(ui_data_queue,), daemon=True)
    t.start()
    
    return display_process

if __name__ == "__main__":
    """
    Allows the UI to be run in a standalone, static mode for layout review.
    This runs the PyQt5 app directly in the main thread.
    No other system components are active. No data is generated or displayed.
    """
    print("--- Standalone UI Layout Review (PyQt5) ---")
    print("Launching ONLY the display UI in a static, non-updating state.")

    # Create dummy queues that will remain empty.
    dummy_data_q = queue.Queue()
    dummy_cmd_q = queue.Queue()

    # Create the QApplication instance.
    app = QtWidgets.QApplication(sys.argv)

    # Create the LiveDisplay window.
    window = LiveDisplay(dummy_data_q, dummy_cmd_q)
    
    # Show the window.
    window.show()

    # Execute the application's event loop.
    sys.exit(app.exec_())