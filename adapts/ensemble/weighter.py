import numpy as np
from collections import deque
from typing import Tuple


class AdapTSWeighter:
    def __init__(self, memory_size: int = 20, beta: float = 0.9):
        self.memory_size = memory_size
        self.beta = beta
        self.fm_residuals = deque(maxlen=memory_size)
        self.adapts_residuals = deque(maxlen=memory_size)
        self.fm_weight = 0.5
        self.adapts_weight = 0.5
        
    def update_residuals(self, y_true: np.ndarray, fm_pred: np.ndarray, adapts_pred: np.ndarray):
        fm_residual = np.mean(np.abs(y_true - fm_pred))
        adapts_residual = np.mean(np.abs(y_true - adapts_pred))
        self.fm_residuals.append(fm_residual)
        self.adapts_residuals.append(adapts_residual)
        self._compute_weights()
    
    def _compute_weights(self):
        if len(self.fm_residuals) < 2:
            self.fm_weight = 0.5
            self.adapts_weight = 0.5
            return
        
        fm_res = np.array(self.fm_residuals)
        adapts_res = np.array(self.adapts_residuals)
        time_weights = np.array([self.beta ** i for i in range(len(fm_res)-1, -1, -1)])
        time_weights /= time_weights.sum()
        
        fm_avg_residual = np.sum(fm_res * time_weights)
        adapts_avg_residual = np.sum(adapts_res * time_weights)
        
        eps = 1e-8
        fm_score = 1.0 / (fm_avg_residual + eps)
        adapts_score = 1.0 / (adapts_avg_residual + eps)
        total_score = fm_score + adapts_score
        
        self.fm_weight = fm_score / total_score
        self.adapts_weight = adapts_score / total_score
    
    def combine_predictions(self, fm_pred: np.ndarray, adapts_pred: np.ndarray) -> np.ndarray:
        combined = self.fm_weight * fm_pred + self.adapts_weight * adapts_pred
        return combined
    
    def get_weights(self) -> Tuple[float, float]:
        return self.fm_weight, self.adapts_weight