import numpy as np
import logging
import pickle
from typing import Dict, List, Any
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetaMind:
    """Meta-level architectural introspection and growth planner.

    Tracks rolling performance indicators (health, prediction error, SOM failures,
    goal satisfaction, reward intensity) and proposes structural growth actions
    once persistent issues are detected and a minimum interval has elapsed.
    """
    def __init__(self, unified_cognitive_state_dim: int | None = None):
        self.history: Dict[str, List[float]] = {
            'health_scores': [], 
            'predict_errors': [], 
            'som_failures': [], 
            'goals_sat': [],
            'reward_intensities': []
        }
        self.growth_threshold = 0.8  # High persistent issues → grow
        self.proposals: List[str] = []  # e.g., "increase_mind_dim:2048"
        
        # Growth tracking
        self.min_growth_interval = 100  # Minimum cycles between growth
        self.last_growth_cycle = 0
        self.current_cycle = 0

        # For future scaling heuristics we may adapt thresholds based on unified state dim
        self.unified_cognitive_state_dim = unified_cognitive_state_dim
        if unified_cognitive_state_dim is not None:
            logger.info(f"MetaMind initialized. Observing unified state dim={unified_cognitive_state_dim}")

    def analyze_trends(self, health_score: float, predict_error: float, som_failures: int, 
                      goals_sat: float, reward_intensity: float):
        """Update history and check for issues."""
        self.current_cycle += 1
        
        # Update history
        self.history['health_scores'].append(health_score)
        self.history['predict_errors'].append(predict_error)
        self.history['som_failures'].append(som_failures)
        self.history['goals_sat'].append(goals_sat)
        self.history['reward_intensities'].append(reward_intensity)

        # Keep history bounded
        max_history = 1000
        for key in self.history:
            if len(self.history[key]) > max_history:
                self.history[key] = self.history[key][-max_history:]

        # Only analyze if enough cycles have passed since last growth
        if self.current_cycle - self.last_growth_cycle < self.min_growth_interval:
            return

        # Analyze recent trends (last 100 cycles)
        window = 100
        recent_errors = self.history['predict_errors'][-window:]
        recent_goals = self.history['goals_sat'][-window:]
        recent_failures = self.history['som_failures'][-window:]

        avg_error = np.mean(recent_errors)
        avg_goals = np.mean(recent_goals)
        avg_failures = np.mean(recent_failures)

        # Check for persistent issues
        if avg_error > 0.4 or avg_goals < 0.6 or avg_failures > 10:
            # Determine appropriate growth
            if avg_error > 0.4:  # High prediction error
                self.proposals.append("increase_mind_dim:2048")
            if avg_failures > 10:  # SOM mapping issues
                self.proposals.append("increase_som_size:31")
            if avg_goals < 0.6:  # Goal satisfaction issues
                self.proposals.append("increase_emotion_dim:2048")
            
            logger.info(f"Growth proposed due to: error={avg_error:.2f}, goals={avg_goals:.2f}, failures={avg_failures:.2f}")
            self.last_growth_cycle = self.current_cycle

    def get_proposals(self) -> List[str]:
        """Get and clear current growth proposals."""
        proposals = self.proposals.copy()
        self.proposals.clear()
        return proposals

    # def save_state(self, path: str):
    #     """Save optimizer state to disk."""
    #     state = {
    #         'history': self.history,
    #         'last_growth_cycle': self.last_growth_cycle,
    #         'current_cycle': self.current_cycle
    #     }
    #     with open(path, 'wb') as f:
    #         pickle.dump(state, f)

    # def load_state(self, path: str):
    #     """Load optimizer state from disk."""
    #     if os.path.exists(path):
    #         with open(path, 'rb') as f:
    #             state = pickle.load(f)
    #             self.history = state['history']
    #             self.last_growth_cycle = state['last_growth_cycle']
    #             self.current_cycle = state['current_cycle']
