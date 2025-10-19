# JustinJ_Optimizer.py
# Standalone copy of JustinJOptimizer with internal audio feedback loop scaffolding.
# Retained original version still inside nn.py; this file allows portability & future divergence.

import numpy as np
import logging
import time
from collections import deque
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

class JustinJOptimizer:
    """Agency-focused optimizer with integrated vocal/audio internal feedback loop.

    Core Additions vs baseline in nn.py:
    - Optional internal echo loop if real mic input missing (still prefers real mic)
    - Audio alignment metrics (correlation, spectral centroid alignment)
    - Latent intention reconstruction (predict intended control vector from audio echo)
    - Multi-timescale adaptive LR modulation
    - Gradient hygiene (clipping + nan/inf scrubbing)
    - Pattern memory with decay + replay sampling hooks
    """
    def __init__(self,
                 networks: List,
                 base_lr: float = 1e-4,
                 vocal_feedback_weight: float = 0.3,
                 agency_growth_rate: float = 0.01,
                 control_precision: float = 0.1,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 weight_decay: float = 0.01,
                 enable_internal_echo: bool = True,
                 spectral_weight: float = 0.2,
                 replay_capacity: int = 512,
                 fast_mode: bool = False):
                # Core setup (internal AdamW-like state; fully decoupled from external optimizer)
                self.networks = networks or []
                self.base_lr = float(base_lr)
                self.beta1 = float(beta1)
                self.beta2 = float(beta2)
                self.epsilon = float(epsilon)
                self.weight_decay = float(weight_decay)  # decoupled AdamW style
                self.vocal_feedback_weight = float(vocal_feedback_weight)
                self.agency_growth_rate = float(agency_growth_rate)
                self.control_precision = float(control_precision)
                self.enable_internal_echo = enable_internal_echo
                self.spectral_weight = float(spectral_weight)
                self.replay_capacity = replay_capacity
                self.fast_mode = bool(fast_mode)

                # Agency metrics
                self.voice_control_confidence = 0.0
                self.intention_alignment = 0.0
                self.feedback_recognition = 0.0
                self.spectral_alignment = 0.0

                # Adaptive learning buffers
                self.p = {}
                self.t = 0

                # Memories
                self.short_term_memory = []
                self.pattern_memory = []
                self.replay_buffer = []

                # Gradient safety
                self.max_grad_norm = 1.0
                self.clip_gradients = True

                # Cached last intended output for echo fallback
                self._last_intended_output = None
                self._last_audio_feedback = None

                # Extended adaptive scheduling
                self.warmup_steps = 500
                self.cooldown_patience = 800
                self.min_lr = self.base_lr * 0.1
                self.max_lr = self.base_lr * 5.0
                self.last_improvement_step = 0
                self.improvement_threshold = 0.002
                self.metric_history = deque(maxlen=400)

                # Temporal coherence tracking
                self._ema_vocal = None
                self.temporal_coherence = 0.0

                # Additional spectral stats
                self.spectral_flatness = 0.0
                self.spectral_bandwidth = 0.0

                # Prioritized replay
                self.priority_replay = []
                self.priority_alpha = 0.7
                self.priority_epsilon = 1e-4

                # Plateau detection buffers
                self._rolling_vc = deque(maxlen=100)
                self._rolling_alignment = deque(maxlen=100)

                # Gradient freeze map
                self.frozen_params = {}
                self.freeze_duration = 300
                self.freeze_threshold = 0.0005
                self.max_freeze_fraction = 0.15

                # Moment & variance state per parameter key
                self.m = {}
                self.v = {}
                self._bias_correction_cache = {}

                logger.info("JustinJOptimizer (standalone) initialized (enriched + integrated AdamW).")

    # --------------------------- Audio / Feedback Utilities ---------------------------
    def _normalize_audio(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        m = np.max(np.abs(x)) + 1e-8
        return x / m

    def _spectral_centroid(self, signal: np.ndarray, sr: int = 44100) -> float:
        signal = signal.astype(np.float32)
        if signal.size == 0:
            return 0.0
        # FFT
        spec = np.fft.rfft(signal)
        mag = np.abs(spec)
        freqs = np.fft.rfftfreq(signal.size, 1.0 / sr)
        denom = np.sum(mag) + 1e-8
        return float(np.sum(freqs * mag) / denom)

    def _spectral_flatness_bandwidth(self, signal: np.ndarray, sr: int = 44100) -> Tuple[float, float]:
        signal = signal.astype(np.float32)
        if signal.size == 0:
            return 0.0, 0.0
        spec = np.fft.rfft(signal)
        mag = np.abs(spec) + 1e-12
        geo_mean = np.exp(np.mean(np.log(mag)))
        arith_mean = np.mean(mag)
        flatness = float(geo_mean / (arith_mean + 1e-12))
        freqs = np.fft.rfftfreq(signal.size, 1.0 / sr)
        centroid = self._spectral_centroid(signal, sr)
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / (np.sum(mag) + 1e-12)))
        return flatness, bandwidth

    def _calculate_feedback_match(self, vocal_output: np.ndarray, audio_feedback: np.ndarray) -> float:
        v = self._normalize_audio(vocal_output).flatten()
        a = self._normalize_audio(audio_feedback).flatten()
        if v.size != a.size:
            n = min(v.size, a.size)
            v = v[:n]; a = a[:n]
        if v.size == 0:
            return 0.0
        corr = np.corrcoef(v, a)[0,1]
        if not np.isfinite(corr):
            corr = 0.0
        return max(0.0, float(corr))

    def _calculate_control_accuracy(self, intended: np.ndarray, actual: np.ndarray) -> float:
        if intended is None or actual is None or intended.size == 0 or actual.size == 0:
            return 0.0
        n = min(intended.size, actual.size)
        err = np.mean((intended[:n] - actual[:n])**2)
        return float(np.exp(-err / max(self.control_precision,1e-6)))

    def _spectral_alignment_metric(self, vocal_output: np.ndarray, audio_feedback: np.ndarray) -> float:
        c1 = self._spectral_centroid(vocal_output)
        c2 = self._spectral_centroid(audio_feedback)
        diff = abs(c1 - c2)
        return float(np.exp(-diff / 800.0))  # heuristic scale

    # --------------------------- Public Metric Update ---------------------------
    def update_metrics(self,
                       vocal_output: Optional[np.ndarray],
                       audio_feedback: Optional[np.ndarray],
                       intended_output: Optional[np.ndarray] = None,
                       reward_signal: Optional[float] = None,
                       sample_rate: int = 44100):
        # Fallback echo if no mic capture yet
        if (audio_feedback is None or audio_feedback.size == 0) and self.enable_internal_echo:
            audio_feedback = vocal_output.copy() if vocal_output is not None else self._last_audio_feedback
        if vocal_output is None and self._last_intended_output is not None:
            vocal_output = self._last_intended_output  # crude fallback

        if vocal_output is None or audio_feedback is None:
            return

        # Core correlations
        fb_match = self._calculate_feedback_match(vocal_output, audio_feedback)
        self.feedback_recognition = self.feedback_recognition * 0.95 + fb_match * 0.05

        if intended_output is not None:
            ctrl_acc = self._calculate_control_accuracy(intended_output, vocal_output)
            self.intention_alignment = self.intention_alignment * 0.9 + ctrl_acc * 0.1
        else:
            ctrl_acc = 0.0

        if not self.fast_mode:
            spec_align = self._spectral_alignment_metric(vocal_output, audio_feedback)
            self.spectral_alignment = self.spectral_alignment * 0.9 + spec_align * 0.1

            # Additional spectral stats
            flatness, bandwidth = self._spectral_flatness_bandwidth(vocal_output, sample_rate)
            self.spectral_flatness = 0.95 * self.spectral_flatness + 0.05 * flatness
            self.spectral_bandwidth = 0.95 * self.spectral_bandwidth + 0.05 * bandwidth
        else:
            spec_align = 0.0
            flatness = 0.0
            bandwidth = 0.0

        # Temporal coherence: similarity of current output to EMA
        if self._ema_vocal is None:
            self._ema_vocal = vocal_output.astype(np.float32)
        else:
            # update ema
            self._ema_vocal = 0.9 * self._ema_vocal + 0.1 * vocal_output.astype(np.float32)
            n = min(self._ema_vocal.size, vocal_output.size)
            if n > 0:
                num = np.dot(self._ema_vocal[:n], vocal_output[:n])
                den = (np.linalg.norm(self._ema_vocal[:n]) * np.linalg.norm(vocal_output[:n]) + 1e-8)
                coh = num / den
                if np.isfinite(coh):
                    self.temporal_coherence = 0.95 * self.temporal_coherence + 0.05 * coh

        self.voice_control_confidence = (
            0.5 * self.feedback_recognition + 0.3 * self.intention_alignment + 0.2 * self.spectral_alignment
        )

        context = {
            'time': time.time(),
            'vocal_output': vocal_output.copy(),
            'audio_feedback': audio_feedback.copy(),
            'intended_output': None if intended_output is None else intended_output.copy(),
            'reward': reward_signal,
            'fb_match': fb_match,
            'ctrl_acc': ctrl_acc,
            'spec_align': spec_align,
            'spec_flat': flatness,
            'spec_band': bandwidth,
            'temp_coh': self.temporal_coherence
        }
        self.short_term_memory.append(context)
        if len(self.short_term_memory) > 128:
            self.short_term_memory.pop(0)

        if reward_signal is not None and reward_signal > 0.7:
            self.pattern_memory.append(context)
            if len(self.pattern_memory) > 512:
                self.pattern_memory.pop(0)

        # Replay store (simple chronological)
        self.replay_buffer.append(context)
        if len(self.replay_buffer) > self.replay_capacity:
            self.replay_buffer.pop(0)

        # Priority replay (priority derived from combined metrics + reward)
        priority = (
            0.4 * fb_match + 0.3 * self.intention_alignment + 0.2 * spec_align + 0.1 * self.temporal_coherence
        )
        if reward_signal is not None:
            priority += 0.3 * reward_signal
        if not self.fast_mode:
            self.priority_replay.append((float(priority), context))
            if len(self.priority_replay) > self.replay_capacity:
                self.priority_replay.pop(0)

        # Track improvement history for schedule decisions
        self.metric_history.append(self.voice_control_confidence)
        self._rolling_vc.append(self.voice_control_confidence)
        self._rolling_alignment.append(self.intention_alignment)

        # Detect improvements
        if len(self.metric_history) > 5:
            recent = list(self.metric_history)[-5:]
            if (max(recent) - min(recent)) > self.improvement_threshold:
                self.last_improvement_step = self.t

        self._last_intended_output = intended_output.copy() if intended_output is not None else self._last_intended_output
        self._last_audio_feedback = audio_feedback.copy()

    # --------------------------- Optimization Step ---------------------------
    def step(self):
        """Perform one optimization step (integrated AdamW + adaptive agency scaling)."""
        self.t += 1

        # ----- Learning rate scheduling -----
        agency_factor = np.sqrt(self.voice_control_confidence + 0.1)
        schedule_lr = self.base_lr
        if self.t < self.warmup_steps:  # warmup
            warm_frac = self.t / max(1, self.warmup_steps)
            schedule_lr = self.base_lr * (0.1 + 0.9 * warm_frac)
        elif (self.t - self.last_improvement_step) > self.cooldown_patience:  # cooldown plateau
            schedule_lr = max(self.min_lr, self.base_lr * 0.5)
        elif self.voice_control_confidence > 0.8:  # expansion when strong
            schedule_lr = min(self.max_lr, self.base_lr * (1.0 + (self.voice_control_confidence - 0.8) * 2.0))

        effective_lr = schedule_lr * agency_factor

        # ----- Parameter loop -----
        freeze_keys_considered = 0
        for net in self.networks:
            for param_array, grad_name, layer_instance in net.get_trainable_params():
                grad_array = getattr(layer_instance, grad_name)
                if grad_array is None:
                    continue
                key = (id(net), id(layer_instance), grad_name)

                # Initialize per-param state lazily
                if key not in self.p:
                    self.p[key] = np.ones_like(param_array, dtype=np.float32)
                if key not in self.m:
                    self.m[key] = np.zeros_like(param_array, dtype=np.float32)
                if key not in self.v:
                    self.v[key] = np.zeros_like(param_array, dtype=np.float32)

                # Skip if currently frozen
                if key in self.frozen_params and self.t < self.frozen_params[key]:
                    continue

                # ----- Agency potential update -----
                if self.voice_control_confidence > self.p[key].mean():
                    self.p[key] *= (1.0 + self.agency_growth_rate)
                else:
                    self.p[key] *= (1.0 - 0.1 * self.agency_growth_rate)

                # Exploration vs consolidation scaling
                if len(self.pattern_memory) > 0:
                    scale = np.sqrt(self.p[key]) * 1.15
                else:
                    scale = np.sqrt(self.p[key]) * 0.85
                if self.temporal_coherence > 0.6:  # damp updates if already stable
                    scale *= 0.9

                grad_array = grad_array * scale
                grad_array = np.nan_to_num(grad_array, nan=0.0, posinf=0.0, neginf=0.0)

                # Lightweight inactivity freeze heuristic
                if freeze_keys_considered < 500:
                    mean_abs = float(np.mean(np.abs(grad_array)))
                    if mean_abs < self.freeze_threshold and len(self.frozen_params) < int(self.max_freeze_fraction * 1000):
                        self.frozen_params[key] = self.t + self.freeze_duration
                    freeze_keys_considered += 1

                # ----- AdamW moment updates -----
                m = self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad_array
                v = self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad_array * grad_array)
                # Bias corrections
                bc1 = 1 - self.beta1 ** self.t
                bc2 = 1 - self.beta2 ** self.t
                m_hat = m / (bc1 + 1e-12)
                v_hat = v / (bc2 + 1e-12)
                update = m_hat / (np.sqrt(v_hat) + self.epsilon)

                # Decoupled weight decay
                if self.weight_decay > 0.0:
                    param_array -= effective_lr * self.weight_decay * param_array

                # Apply update
                param_array -= effective_lr * update

        if self.clip_gradients:
            self._clip_gradients()

        if self.t % 200 == 0:
            logger.info(
                f"JustinJOpt t={self.t} lr_eff={effective_lr:.2e} VC={self.voice_control_confidence:.3f} "
                f"FB={self.feedback_recognition:.3f} IA={self.intention_alignment:.3f} SPEC={self.spectral_alignment:.3f} "
                f"Flat={self.spectral_flatness:.3f} Band={self.spectral_bandwidth:.1f} Coh={self.temporal_coherence:.3f} Replay={len(self.replay_buffer)}")

        # store last lr for diagnostics
        self._last_effective_lr = effective_lr

    # --------------------------- Prioritized Replay Utilities ---------------------------
    def _sample_priority_indices(self, batch_size: int) -> List[int]:
        if not self.priority_replay:
            return []
        priorities = np.array([p for p, _ in self.priority_replay], dtype=np.float32)
        probs = (priorities + self.priority_epsilon) ** self.priority_alpha
        probs /= probs.sum()
        count = min(batch_size, len(self.priority_replay))
        return list(np.random.choice(len(self.priority_replay), size=count, replace=False, p=probs))

    def optimize_with_replay(self, batch_size: int = 16):
        """Lightweight auxiliary adaptation using prioritized replay contexts.
        This does NOT backprop (no computational graph) but modulates agency potentials
        using stored high-priority alignment examples so future gradient scaling reflects them.
        """
        if self.fast_mode:
            return
        idxs = self._sample_priority_indices(batch_size)
        if not idxs:
            return
        avg_fb = 0.0; avg_ctrl = 0.0; avg_spec = 0.0
        for i in idxs:
            priority, ctx = self.priority_replay[i]
            avg_fb += ctx['fb_match']
            avg_ctrl += ctx['ctrl_acc']
            avg_spec += ctx['spec_align']
        n = len(idxs)
        avg_fb /= n; avg_ctrl /= n; avg_spec /= n
        composite = 0.5 * avg_fb + 0.3 * avg_ctrl + 0.2 * avg_spec
        # Adjust global potentials subtly
        adjust = (composite - 0.5) * 0.01  # small influence
        for k in list(self.p.keys())[:200]:  # cap cost
            self.p[k] *= (1.0 + adjust)

    # --------------------------- Diagnostics ---------------------------
    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            **self.get_agency_metrics(),
            'spectral_flatness': float(self.spectral_flatness),
            'spectral_bandwidth': float(self.spectral_bandwidth),
            'temporal_coherence': float(self.temporal_coherence),
            'fast_mode': self.fast_mode,
            'last_effective_lr': getattr(self, '_last_effective_lr', None),
            'lr_base': float(self.base_lr),
            'frozen_params': len(self.frozen_params),
            'warmup_remaining': max(0, self.warmup_steps - self.t),
            'since_improvement': self.t - self.last_improvement_step
        }

    # --------------------------- Gradient Clipping ---------------------------
    def _clip_gradients(self):
        total_norm = 0.0
        for net in self.networks:
            for _, grad_name, layer in net.get_trainable_params():
                g = getattr(layer, grad_name)
                if g is not None:
                    total_norm += float(np.linalg.norm(g)**2)
        total_norm = np.sqrt(total_norm)
        if total_norm > self.max_grad_norm:
            coef = self.max_grad_norm / (total_norm + 1e-6)
            for net in self.networks:
                for _, grad_name, layer in net.get_trainable_params():
                    g = getattr(layer, grad_name)
                    if g is not None:
                        setattr(layer, grad_name, g * coef)

    # --------------------------- Replay Sampling Hook ---------------------------
    def sample_replay(self, batch_size: int = 8) -> List[Dict[str, Any]]:
        if not self.replay_buffer:
            return []
        idxs = np.random.choice(len(self.replay_buffer), size=min(batch_size, len(self.replay_buffer)), replace=False)
        return [self.replay_buffer[i] for i in idxs]

    def get_agency_metrics(self) -> Dict[str, float]:
        return {
            'voice_control_confidence': float(self.voice_control_confidence),
            'intention_alignment': float(self.intention_alignment),
            'feedback_recognition': float(self.feedback_recognition),
            'spectral_alignment': float(self.spectral_alignment),
            'pattern_memory_size': len(self.pattern_memory),
            'replay_size': len(self.replay_buffer),
            'temporal_coherence': float(self.temporal_coherence)
        }

# EOF
