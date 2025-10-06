import numpy as np
from numpy.fft import fft
from sklearn.linear_model import Ridge
from collections import deque


class AdapTSForecaster:
    def __init__(self, alpha: float = 1.0, window_size: int = 20, pred_len: int = 24):
        self.alpha = alpha
        self.window_size = window_size
        self.pred_len = pred_len
        self.model = Ridge(alpha=alpha)
        self.window = deque(maxlen=window_size)
        self.is_fitted = False
        
    def transform_input(self, x: np.ndarray) -> np.ndarray:
        fft_vals = fft(x)
        features = np.abs(fft_vals[:len(x)//2])
        return features
    
    def update(self, x: np.ndarray, y: np.ndarray):
        x_transformed = self.transform_input(x)
        if np.isscalar(y):
            y = np.array([y])
        elif isinstance(y, list):
            y = np.array(y)
        self.window.append((x_transformed, y))
        if len(self.window) >= min(5, self.window_size):
            self._refit()
    
    def _refit(self):
        if len(self.window) == 0:
            return
        X = np.vstack([feat for feat, _ in self.window])
        y = np.vstack([targ.reshape(1, -1) for _, targ in self.window])
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.full(self.pred_len, x[-1])
        x_transformed = self.transform_input(x)
        prediction = self.model.predict(x_transformed.reshape(1, -1))
        return prediction.flatten()