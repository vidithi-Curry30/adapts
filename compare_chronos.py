"""
Compare Chronos (Amazon's pre-trained FM) with and without AdapTS
Tests: Chronos + Conformal vs Chronos + AdapTS + Conformal
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from chronos import ChronosPipeline
from adapts.data import TimeSeriesDataLoader
from adapts.models import AdapTSForecaster
from adapts.ensemble import AdapTSWeighter
from adapts.uncertainty import ConformalPredictor

print("="*70)
print("Comparing Chronos (Pre-trained FM) with/without AdapTS")
print("="*70)

# ============================================================================
# STEP 1: Load Chronos (Pre-trained Foundation Model)
# ============================================================================

print("\n[1] Loading Chronos (Amazon's pre-trained time series model)...")

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny",
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)

print("✓ Chronos loaded successfully")

class ChronosWrapper:
    """Wrapper to make Chronos compatible with our API"""
    def __init__(self, pipeline, pred_len=24):
        self.pipeline = pipeline
        self.pred_len = pred_len
    
    def predict(self, x):
        """Make prediction compatible with our interface"""
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        # Chronos expects list of 1D tensors
        context = [torch.tensor(x[0], dtype=torch.float32)]
        
        # Generate forecast
        forecast = self.pipeline.predict(context, self.pred_len)
        
        # Return median prediction
        pred = forecast[0].median(dim=0).values.numpy()
        return pred.reshape(1, -1)

chronos_fm = ChronosWrapper(pipeline, pred_len=24)

# Test it works
print("Testing Chronos prediction...")
test_input = np.random.randn(96)
test_output = chronos_fm.predict(test_input)
print(f"✓ Test successful: input shape {test_input.shape} → output shape {test_output.shape}")

# ============================================================================
# STEP 2: Test on Stock Datasets
# ============================================================================

TEST_TICKERS = ['GOOGL', 'MSFT', 'WMT']  # Start with 3 for speed
all_results = {}

for ticker in TEST_TICKERS:
    print(f"\n{'='*70}")
    print(f"Testing on {ticker}")
    print(f"{'='*70}")
    
    # Load data
    loader = TimeSeriesDataLoader(source='stock')
    train_data, test_data = loader.load(ticker=ticker, start='2020-01-01', end='2024-01-01')
    test_data = test_data[:150]  # Small subset for speed
    
    seq_len = 96
    pred_len = 24
    
    # ========================================================================
    # METHOD 1: Chronos + Conformal (Zero-shot)
    # ========================================================================
    print("\n[1] Chronos + Conformal (zero-shot)...")
    
    chronos_conf = ConformalPredictor(alpha=0.1, calibration_size=50)
    chronos_coverages = []
    chronos_widths = []
    chronos_errors = []
    
    n_steps = len(test_data) - seq_len - pred_len
    for i in range(n_steps):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{n_steps}")
        
        x = test_data[i:i+seq_len]
        y_true = test_data[i+seq_len:i+seq_len+pred_len]
        
        # Chronos prediction
        pred = chronos_fm.predict(x).flatten()
        
        # Conformal interval
        result = chronos_conf.predict_with_interval(pred)
        chronos_conf.update_residuals(y_true, pred)
        
        # Metrics
        covered = sum(1 for j in range(len(y_true)) 
                     if result['lower'][j] <= y_true[j] <= result['upper'][j])
        chronos_coverages.append(covered / len(y_true))
        chronos_widths.append(np.mean(result['upper'] - result['lower']))
        chronos_errors.append(np.mean(np.abs(y_true - pred)))
    
    print(f"   Avg Coverage: {np.mean(chronos_coverages)*100:.1f}%")
    print(f"   Avg MAE: {np.mean(chronos_errors):.4f}")
    
    # ========================================================================
    # METHOD 2: Chronos + AdapTS + Conformal
    # ========================================================================
    print("\n[2] Chronos + AdapTS + Conformal...")
    
    forecaster = AdapTSForecaster(alpha=0.1, window_size=5, pred_len=24)
    weighter = AdapTSWeighter(memory_size=20, beta=0.9)
    conf = ConformalPredictor(alpha=0.1, calibration_size=50)
    
    adapts_coverages = []
    adapts_widths = []
    adapts_errors = []
    
    for i in range(n_steps):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{n_steps}")
        
        x = test_data[i:i+seq_len]
        y_true = test_data[i+seq_len:i+seq_len+pred_len]
        
        # Both predictions
        chronos_pred = chronos_fm.predict(x).flatten()
        local_pred = forecaster.predict(x)
        combined_pred = weighter.combine_predictions(chronos_pred, local_pred)
        
        # Conformal interval
        result = conf.predict_with_interval(combined_pred)
        
        # Update
        forecaster.update(x, y_true)
        weighter.update_residuals(y_true, chronos_pred, local_pred)
        conf.update_residuals(y_true, combined_pred)
        
        # Metrics
        covered = sum(1 for j in range(len(y_true)) 
                     if result['lower'][j] <= y_true[j] <= result['upper'][j])
        adapts_coverages.append(covered / len(y_true))
        adapts_widths.append(np.mean(result['upper'] - result['lower']))
        adapts_errors.append(np.mean(np.abs(y_true - combined_pred)))
    
    print(f"   Avg Coverage: {np.mean(adapts_coverages)*100:.1f}%")
    print(f"   Avg MAE: {np.mean(adapts_errors):.4f}")
    
    improvement = (np.mean(chronos_errors) - np.mean(adapts_errors)) / np.mean(chronos_errors) * 100
    print(f"   ✓ Improvement: {improvement:.1f}%")
    
    all_results[ticker] = {
        'chronos': {'avg_mae': np.mean(chronos_errors), 'avg_coverage': np.mean(chronos_coverages)},
        'chronos_adapts': {'avg_mae': np.mean(adapts_errors), 'avg_coverage': np.mean(adapts_coverages)},
        'improvement': improvement
    }

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}\n")

for ticker in TEST_TICKERS:
    res = all_results[ticker]
    print(f"{ticker}:")
    print(f"  Chronos alone: MAE={res['chronos']['avg_mae']:.4f}, Coverage={res['chronos']['avg_coverage']*100:.1f}%")
    print(f"  With AdapTS:   MAE={res['chronos_adapts']['avg_mae']:.4f}, Coverage={res['chronos_adapts']['avg_coverage']*100:.1f}%")
    print(f"  Improvement:   {res['improvement']:+.1f}%\n")

avg_improvement = np.mean([all_results[t]['improvement'] for t in TEST_TICKERS])
print(f"Average improvement: {avg_improvement:+.1f}%")
print("\n✓ Complete!")