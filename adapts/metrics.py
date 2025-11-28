"""
Evaluation metrics for time series forecasting.

This module provides standardized metrics for evaluating forecast accuracy
and uncertainty quantification performance.
"""

import numpy as np
from typing import Dict, List, Tuple, Union


class TimeSeriesMetrics:
    """Collection of time series forecasting evaluation metrics."""

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error.

        Args:
            y_true: Ground truth values (shape: (n_samples,) or (n_sequences, pred_len))
            y_pred: Predicted values (same shape as y_true)

        Returns:
            Mean absolute error as a scalar
        """
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error.

        Args:
            y_true: Ground truth values (shape: (n_samples,) or (n_sequences, pred_len))
            y_pred: Predicted values (same shape as y_true)

        Returns:
            Root mean squared error as a scalar
        """
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Squared Error.

        Args:
            y_true: Ground truth values (shape: (n_samples,) or (n_sequences, pred_len))
            y_pred: Predicted values (same shape as y_true)

        Returns:
            Mean squared error as a scalar
        """
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
        """
        Calculate Mean Absolute Percentage Error.

        Args:
            y_true: Ground truth values (shape: (n_samples,) or (n_sequences, pred_len))
            y_pred: Predicted values (same shape as y_true)
            epsilon: Small value to avoid division by zero

        Returns:
            Mean absolute percentage error as a percentage
        """
        return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100)

    @staticmethod
    def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
        """
        Calculate empirical coverage of prediction intervals.

        Args:
            y_true: Ground truth values (shape: (pred_len,))
            lower: Lower bounds of prediction intervals (same shape as y_true)
            upper: Upper bounds of prediction intervals (same shape as y_true)

        Returns:
            Coverage ratio (between 0 and 1)
        """
        covered = np.sum((lower <= y_true) & (y_true <= upper))
        return float(covered / len(y_true))

    @staticmethod
    def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
        """
        Calculate average width of prediction intervals.

        Args:
            lower: Lower bounds of prediction intervals
            upper: Upper bounds of prediction intervals

        Returns:
            Average interval width
        """
        return float(np.mean(upper - lower))

    @staticmethod
    def compute_all(y_true: np.ndarray, y_pred: np.ndarray,
                   lower: np.ndarray = None, upper: np.ndarray = None) -> Dict[str, float]:
        """
        Compute all available metrics.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            lower: Optional lower bounds for interval metrics
            upper: Optional upper bounds for interval metrics

        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {
            'mae': TimeSeriesMetrics.mae(y_true, y_pred),
            'rmse': TimeSeriesMetrics.rmse(y_true, y_pred),
            'mse': TimeSeriesMetrics.mse(y_true, y_pred),
            'mape': TimeSeriesMetrics.mape(y_true, y_pred),
        }

        if lower is not None and upper is not None:
            metrics['coverage'] = TimeSeriesMetrics.coverage(y_true, lower, upper)
            metrics['interval_width'] = TimeSeriesMetrics.interval_width(lower, upper)

        return metrics


def evaluate_forecasts(true_values: List[np.ndarray],
                      predictions: List[Dict[str, np.ndarray]]) -> Dict[str, float]:
    """
    Evaluate a list of forecasts with comprehensive metrics.

    Args:
        true_values: List of ground truth arrays
        predictions: List of prediction dictionaries containing:
            - 'prediction': point forecasts
            - 'lower': (optional) lower bounds
            - 'upper': (optional) upper bounds

    Returns:
        Dictionary of averaged metrics across all forecasts
    """
    mae_values = []
    rmse_values = []
    coverage_values = []
    width_values = []

    for y_true, pred_dict in zip(true_values, predictions):
        y_pred = pred_dict['prediction']

        # Point forecast metrics
        mae_values.append(TimeSeriesMetrics.mae(y_true, y_pred))
        rmse_values.append(TimeSeriesMetrics.rmse(y_true, y_pred))

        # Interval metrics (if available)
        if 'lower' in pred_dict and 'upper' in pred_dict:
            coverage_values.append(TimeSeriesMetrics.coverage(
                y_true, pred_dict['lower'], pred_dict['upper']
            ))
            width_values.append(TimeSeriesMetrics.interval_width(
                pred_dict['lower'], pred_dict['upper']
            ))

    results = {
        'mae': float(np.mean(mae_values)),
        'rmse': float(np.mean(rmse_values)),
        'mae_std': float(np.std(mae_values)),
        'rmse_std': float(np.std(rmse_values)),
    }

    if coverage_values:
        results['coverage'] = float(np.mean(coverage_values))
        results['coverage_std'] = float(np.std(coverage_values))
        results['interval_width'] = float(np.mean(width_values))
        results['interval_width_std'] = float(np.std(width_values))

<<<<<<< HEAD
    return results
=======
    return results
>>>>>>> 2c5b864d712ab2174962d6cbbc8585b86a6a6c04
