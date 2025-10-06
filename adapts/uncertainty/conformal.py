import numpy as np
from collections import deque
from typing import Tuple


class ConformalPredictor:
    def __init__(self, alpha: float = 0.1, calibration_size: int = 50):
        self.alpha = alpha
        self.calibration_size = calibration_size
        self.residuals = deque(maxlen=calibration_size)
        
    def update_residuals(self, y_true: np.ndarray, y_pred: np.ndarray):
        residuals = np.abs(y_true - y_pred)
        for r in residuals:
            self.residuals.append(r)
    
    def get_quantile(self) -> float:
        if len(self.residuals) < 5:
            return 1.0
        residuals_array = np.array(self.residuals)
        quantile = np.quantile(residuals_array, 1 - self.alpha)
        return quantile
    
    def predict_interval(self, point_forecast: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        quantile = self.get_quantile()
        lower = point_forecast - quantile
        upper = point_forecast + quantile
        return lower, upper
    
    def predict_with_interval(self, point_forecast: np.ndarray) -> dict:
        lower, upper = self.predict_interval(point_forecast)
        return {
            'prediction': point_forecast,
            'lower': lower,
            'upper': upper,
            'quantile': self.get_quantile()
        }