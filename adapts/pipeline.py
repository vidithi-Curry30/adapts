import numpy as np
from typing import Dict, Optional
from .models.forecaster import AdapTSForecaster
from .ensemble.weighter import AdapTSWeighter
from .uncertainty.conformal import ConformalPredictor


class AdapTS:
    def __init__(self, foundation_model, seq_len: int = 96, pred_len: int = 24,
                 alpha_ridge: float = 1.0, window_size: int = 20, memory_size: int = 20,
                 beta: float = 0.9, conf_alpha: float = 0.1, calibration_size: int = 50):
        self.fm = foundation_model
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.forecaster = AdapTSForecaster(alpha_ridge, window_size, pred_len)
        self.weighter = AdapTSWeighter(memory_size, beta)
        self.conformal = ConformalPredictor(conf_alpha, calibration_size)
        
        self.metrics = {
            'fm_weights': [],
            'adapts_weights': [],
            'errors': []
        }
    
    def predict(self, x: np.ndarray, return_components: bool = False) -> Dict:
        if len(x) != self.seq_len:
            raise ValueError(f"Input length {len(x)} != seq_len {self.seq_len}")
        
        fm_pred = self.fm.predict(x.reshape(1, -1)).flatten()
        adapts_pred = self.forecaster.predict(x)
        combined_pred = self.weighter.combine_predictions(fm_pred, adapts_pred)
        result = self.conformal.predict_with_interval(combined_pred)
        
        if return_components:
            result['fm_prediction'] = fm_pred
            result['adapts_prediction'] = adapts_pred
            result['fm_weight'], result['adapts_weight'] = self.weighter.get_weights()
        
        return result
    
    def update(self, x: np.ndarray, y_true: np.ndarray):
        fm_pred = self.fm.predict(x.reshape(1, -1)).flatten()
        adapts_pred = self.forecaster.predict(x)
        combined_pred = self.weighter.combine_predictions(fm_pred, adapts_pred)
        
        self.forecaster.update(x, y_true)
        self.weighter.update_residuals(y_true, fm_pred, adapts_pred)
        self.conformal.update_residuals(y_true, combined_pred)
        
        self.metrics['fm_weights'].append(self.weighter.fm_weight)
        self.metrics['adapts_weights'].append(self.weighter.adapts_weight)
        self.metrics['errors'].append(np.mean(np.abs(y_true - combined_pred)))
    
    def evaluate_online(self, test_data: np.ndarray, verbose: bool = True):
        predictions = []
        true_values = []
        n_steps = len(test_data) - self.seq_len - self.pred_len + 1
        
        if verbose:
            print(f"Running online evaluation for {n_steps} steps...")
        
        for i in range(n_steps):
            x = test_data[i:i + self.seq_len]
            y_true = test_data[i + self.seq_len:i + self.seq_len + self.pred_len]
            pred_dict = self.predict(x, return_components=True)
            predictions.append(pred_dict)
            true_values.append(y_true)
            self.update(x, y_true)
            
            if verbose and (i + 1) % 20 == 0:
                print(f"  Step {i+1}/{n_steps}")
        
        mae = np.mean([np.mean(np.abs(t - p['prediction'])) 
                      for t, p in zip(true_values, predictions)])
        coverage = self._compute_coverage(true_values, predictions)
        
        if verbose:
            print(f"\nResults:")
            print(f"  MAE: {mae:.4f}")
            print(f"  Coverage: {coverage*100:.1f}%")
        
        return {'mae': mae, 'coverage': coverage, 'predictions': predictions, 'true_values': true_values}
    
    def _compute_coverage(self, true_values, predictions):
        covered = 0
        for y_true, pred in zip(true_values, predictions):
            within = np.all((y_true >= pred['lower']) & (y_true <= pred['upper']))
            if within:
                covered += 1
        return covered / len(true_values) if len(true_values) > 0 else 0.0