"""
Comprehensive Conformal Prediction Analysis for AdapTs

This script analyzes how AdapTs forecasts work with conformal prediction in different configurations:
1. Chronos + Conformal (FM with conformal prediction, no AdapTS adaptation)
2. AdapTS + Conformal (Local adaptive forecasting only, no FM)
3. Chronos + AdapTS + Conformal (Full system combining FM with AdapTS adaptation)

Using Amazon Chronos as the foundation model - a model NOT trained directly on stock data.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from chronos import ChronosPipeline
from adapts.data import TimeSeriesDataLoader
from adapts.models import AdapTSForecaster
from adapts.ensemble import AdapTSWeighter
from adapts.uncertainty import ConformalPredictor
from adapts.metrics import TimeSeriesMetrics

print("="*80)
print("COMPREHENSIVE CONFORMAL PREDICTION ANALYSIS FOR ADAPTS")
print("="*80)
print("\nAnalyzing three configurations:")
print("  1. Chronos + Conformal (Zero-shot FM)")
print("  2. AdapTS + Conformal (Local adaptation only, no FM)")
print("  3. Chronos + AdapTS + Conformal (Full system)")
print("\nFoundation Model: Amazon Chronos (NOT trained on stock data)")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

TEST_TICKERS = ['GOOGL', 'MSFT', 'WMT', 'COST', 'USO', 'CVX', 'UNH']
SEQ_LEN = 96
PRED_LEN = 24
CONF_ALPHA = 0.1  # 90% prediction intervals
CALIBRATION_SIZE = 200
MAX_STEPS = 300  # Limit to 300 steps for faster execution (set to None for all data)

# ============================================================================
# STEP 1: Load Chronos Foundation Model
# ============================================================================

print("\n[STEP 1] Loading Chronos Foundation Model...")
print("-" * 80)

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny",
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)

print("✓ Chronos loaded successfully")
print("  Model: amazon/chronos-t5-tiny")
print("  Note: This is a general-purpose time series FM, NOT trained on stock data")

class ChronosWrapper:
    """Wrapper to make Chronos compatible with AdapTS API"""
    def __init__(self, pipeline, pred_len=24):
        self.pipeline = pipeline
        self.pred_len = pred_len

    def predict(self, x):
        """Make prediction compatible with our interface"""
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        # Chronos expects list of 1D tensors
        context = [torch.tensor(x[0], dtype=torch.float32)]

        # Generate forecast (returns samples)
        forecast = self.pipeline.predict(context, self.pred_len)

        # Return median prediction (robust to outliers)
        pred = forecast[0].median(dim=0).values.numpy()
        return pred.reshape(1, -1)

chronos_fm = ChronosWrapper(pipeline, pred_len=PRED_LEN)

# Test prediction
print("\nTesting Chronos prediction...")
test_input = np.random.randn(SEQ_LEN)
test_output = chronos_fm.predict(test_input)
print(f"✓ Test successful: {test_input.shape} → {test_output.shape}")

# ============================================================================
# STEP 2: Run Experiments on All Datasets
# ============================================================================

print(f"\n[STEP 2] Running Experiments on {len(TEST_TICKERS)} Stock Datasets")
print("-" * 80)

all_results = {}

for ticker in TEST_TICKERS:
    print(f"\n{'='*80}")
    print(f"Dataset: {ticker}")
    print(f"{'='*80}")

    # Load data (use all data for evaluation with split_ratio=0.0)
    loader = TimeSeriesDataLoader(source='stock', split_ratio=0.0)
    train_data, test_data = loader.load(
        ticker=ticker,
        start='2015-01-01',
        end='2024-01-01'
    )

    print(f"Loaded {len(test_data)} timesteps")
    n_steps = len(test_data) - SEQ_LEN - PRED_LEN
    if MAX_STEPS is not None:
        n_steps = min(n_steps, MAX_STEPS)
    print(f"Will evaluate {n_steps} forecasting steps")

    # ========================================================================
    # METHOD 1: Chronos + Conformal (Zero-shot, no adaptation)
    # ========================================================================
    print(f"\n[Method 1/3] Chronos + Conformal (Zero-shot)")
    print("  - Foundation model predictions only")
    print("  - No local adaptation")
    print("  - Conformal prediction for uncertainty quantification")

    chronos_conf = ConformalPredictor(alpha=CONF_ALPHA, calibration_size=CALIBRATION_SIZE)
    chronos_results = {
        'coverages': [],
        'widths': [],
        'mae_list': [],
        'rmse_list': []
    }

    for i in range(n_steps):
        x = test_data[i:i+SEQ_LEN]
        y_true = test_data[i+SEQ_LEN:i+SEQ_LEN+PRED_LEN]

        # Chronos prediction
        pred = chronos_fm.predict(x).flatten()

        # Add conformal interval
        result = chronos_conf.predict_with_interval(pred)
        chronos_conf.update_residuals(y_true, pred)

        # Compute metrics
        chronos_results['coverages'].append(
            TimeSeriesMetrics.coverage(y_true, result['lower'], result['upper'])
        )
        chronos_results['widths'].append(
            TimeSeriesMetrics.interval_width(result['lower'], result['upper'])
        )
        chronos_results['mae_list'].append(TimeSeriesMetrics.mae(y_true, pred))
        chronos_results['rmse_list'].append(TimeSeriesMetrics.rmse(y_true, pred))

    # Compute averages
    chronos_results['avg_coverage'] = np.mean(chronos_results['coverages'])
    chronos_results['avg_width'] = np.mean(chronos_results['widths'])
    chronos_results['avg_mae'] = np.mean(chronos_results['mae_list'])
    chronos_results['avg_rmse'] = np.mean(chronos_results['rmse_list'])

    print(f"  ✓ Avg Coverage: {chronos_results['avg_coverage']*100:.1f}%")
    print(f"  ✓ Avg Width: {chronos_results['avg_width']:.4f}")
    print(f"  ✓ Avg MAE: {chronos_results['avg_mae']:.4f}")
    print(f"  ✓ Avg RMSE: {chronos_results['avg_rmse']:.4f}")

    # ========================================================================
    # METHOD 2: AdapTS + Conformal (Local adaptation only, no FM)
    # ========================================================================
    print(f"\n[Method 2/3] AdapTS + Conformal (Local adaptation only)")
    print("  - NO foundation model")
    print("  - Local adaptive forecasting using Ridge regression + FFT features")
    print("  - Online learning with sliding window")
    print("  - Conformal prediction for uncertainty quantification")

    forecaster = AdapTSForecaster(alpha=0.1, window_size=5, pred_len=PRED_LEN)
    adapts_conf = ConformalPredictor(alpha=CONF_ALPHA, calibration_size=CALIBRATION_SIZE)
    adapts_results = {
        'coverages': [],
        'widths': [],
        'mae_list': [],
        'rmse_list': []
    }

    for i in range(n_steps):
        x = test_data[i:i+SEQ_LEN]
        y_true = test_data[i+SEQ_LEN:i+SEQ_LEN+PRED_LEN]

        # Local adaptive prediction
        pred = forecaster.predict(x)

        # Add conformal interval
        result = adapts_conf.predict_with_interval(pred)

        # Update both components
        forecaster.update(x, y_true)
        adapts_conf.update_residuals(y_true, pred)

        # Compute metrics
        adapts_results['coverages'].append(
            TimeSeriesMetrics.coverage(y_true, result['lower'], result['upper'])
        )
        adapts_results['widths'].append(
            TimeSeriesMetrics.interval_width(result['lower'], result['upper'])
        )
        adapts_results['mae_list'].append(TimeSeriesMetrics.mae(y_true, pred))
        adapts_results['rmse_list'].append(TimeSeriesMetrics.rmse(y_true, pred))

    # Compute averages
    adapts_results['avg_coverage'] = np.mean(adapts_results['coverages'])
    adapts_results['avg_width'] = np.mean(adapts_results['widths'])
    adapts_results['avg_mae'] = np.mean(adapts_results['mae_list'])
    adapts_results['avg_rmse'] = np.mean(adapts_results['rmse_list'])

    print(f"  ✓ Avg Coverage: {adapts_results['avg_coverage']*100:.1f}%")
    print(f"  ✓ Avg Width: {adapts_results['avg_width']:.4f}")
    print(f"  ✓ Avg MAE: {adapts_results['avg_mae']:.4f}")
    print(f"  ✓ Avg RMSE: {adapts_results['avg_rmse']:.4f}")

    # ========================================================================
    # METHOD 3: Chronos + AdapTS + Conformal (Full system)
    # ========================================================================
    print(f"\n[Method 3/3] Chronos + AdapTS + Conformal (Full system)")
    print("  - Foundation model (Chronos) predictions")
    print("  - Local adaptive forecasting")
    print("  - Dynamic ensemble weighting (adapts based on recent performance)")
    print("  - Conformal prediction for uncertainty quantification")

    forecaster_full = AdapTSForecaster(alpha=0.1, window_size=5, pred_len=PRED_LEN)
    weighter = AdapTSWeighter(memory_size=20, beta=0.9)
    full_conf = ConformalPredictor(alpha=CONF_ALPHA, calibration_size=CALIBRATION_SIZE)
    full_results = {
        'coverages': [],
        'widths': [],
        'mae_list': [],
        'rmse_list': [],
        'fm_weights': [],
        'adapts_weights': []
    }

    for i in range(n_steps):
        x = test_data[i:i+SEQ_LEN]
        y_true = test_data[i+SEQ_LEN:i+SEQ_LEN+PRED_LEN]

        # Both predictions
        chronos_pred = chronos_fm.predict(x).flatten()
        local_pred = forecaster_full.predict(x)

        # Weighted combination
        combined_pred = weighter.combine_predictions(chronos_pred, local_pred)

        # Store weights for analysis
        weights = weighter.get_weights()
        full_results['fm_weights'].append(weights[0])
        full_results['adapts_weights'].append(weights[1])

        # Add conformal interval
        result = full_conf.predict_with_interval(combined_pred)

        # Update all components
        forecaster_full.update(x, y_true)
        weighter.update_residuals(y_true, chronos_pred, local_pred)
        full_conf.update_residuals(y_true, combined_pred)

        # Compute metrics
        full_results['coverages'].append(
            TimeSeriesMetrics.coverage(y_true, result['lower'], result['upper'])
        )
        full_results['widths'].append(
            TimeSeriesMetrics.interval_width(result['lower'], result['upper'])
        )
        full_results['mae_list'].append(TimeSeriesMetrics.mae(y_true, combined_pred))
        full_results['rmse_list'].append(TimeSeriesMetrics.rmse(y_true, combined_pred))

    # Compute averages
    full_results['avg_coverage'] = np.mean(full_results['coverages'])
    full_results['avg_width'] = np.mean(full_results['widths'])
    full_results['avg_mae'] = np.mean(full_results['mae_list'])
    full_results['avg_rmse'] = np.mean(full_results['rmse_list'])
    full_results['avg_fm_weight'] = np.mean(full_results['fm_weights'])
    full_results['avg_adapts_weight'] = np.mean(full_results['adapts_weights'])

    print(f"  ✓ Avg Coverage: {full_results['avg_coverage']*100:.1f}%")
    print(f"  ✓ Avg Width: {full_results['avg_width']:.4f}")
    print(f"  ✓ Avg MAE: {full_results['avg_mae']:.4f}")
    print(f"  ✓ Avg RMSE: {full_results['avg_rmse']:.4f}")
    print(f"  ✓ Avg FM Weight: {full_results['avg_fm_weight']:.3f}")
    print(f"  ✓ Avg AdapTS Weight: {full_results['avg_adapts_weight']:.3f}")

    # ========================================================================
    # Compute Improvements
    # ========================================================================
    print(f"\n{'─'*80}")
    print("Performance Comparison:")
    print(f"{'─'*80}")

    # MAE improvements
    mae_improve_adapts = (chronos_results['avg_mae'] - adapts_results['avg_mae']) / chronos_results['avg_mae'] * 100
    mae_improve_full = (chronos_results['avg_mae'] - full_results['avg_mae']) / chronos_results['avg_mae'] * 100

    print(f"MAE Improvement vs Chronos-only:")
    print(f"  AdapTS-only: {mae_improve_adapts:+.1f}%")
    print(f"  Full system: {mae_improve_full:+.1f}%")

    # RMSE improvements
    rmse_improve_adapts = (chronos_results['avg_rmse'] - adapts_results['avg_rmse']) / chronos_results['avg_rmse'] * 100
    rmse_improve_full = (chronos_results['avg_rmse'] - full_results['avg_rmse']) / chronos_results['avg_rmse'] * 100

    print(f"\nRMSE Improvement vs Chronos-only:")
    print(f"  AdapTS-only: {rmse_improve_adapts:+.1f}%")
    print(f"  Full system: {rmse_improve_full:+.1f}%")

    # Store all results
    all_results[ticker] = {
        'chronos_conf': chronos_results,
        'adapts_conf': adapts_results,
        'full_system': full_results,
        'improvements': {
            'mae_adapts': mae_improve_adapts,
            'mae_full': mae_improve_full,
            'rmse_adapts': rmse_improve_adapts,
            'rmse_full': rmse_improve_full
        }
    }

# ============================================================================
# STEP 3: Save Results to JSON
# ============================================================================

print(f"\n{'='*80}")
print("[STEP 3] Saving Results")
print("-" * 80)

# Convert results to JSON-serializable format
results_json = {}
for ticker, res in all_results.items():
    results_json[ticker] = {
        'chronos_conf': {
            'avg_coverage': float(res['chronos_conf']['avg_coverage']),
            'avg_width': float(res['chronos_conf']['avg_width']),
            'avg_mae': float(res['chronos_conf']['avg_mae']),
            'avg_rmse': float(res['chronos_conf']['avg_rmse'])
        },
        'adapts_conf': {
            'avg_coverage': float(res['adapts_conf']['avg_coverage']),
            'avg_width': float(res['adapts_conf']['avg_width']),
            'avg_mae': float(res['adapts_conf']['avg_mae']),
            'avg_rmse': float(res['adapts_conf']['avg_rmse'])
        },
        'full_system': {
            'avg_coverage': float(res['full_system']['avg_coverage']),
            'avg_width': float(res['full_system']['avg_width']),
            'avg_mae': float(res['full_system']['avg_mae']),
            'avg_rmse': float(res['full_system']['avg_rmse']),
            'avg_fm_weight': float(res['full_system']['avg_fm_weight']),
            'avg_adapts_weight': float(res['full_system']['avg_adapts_weight'])
        },
        'improvements': res['improvements']
    }

# Save with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f'conformal_analysis_results_{timestamp}.json'

with open(results_file, 'w') as f:
    json.dump(results_json, f, indent=2)

print(f"✓ Results saved to: {results_file}")

# ============================================================================
# STEP 4: Generate Comprehensive Visualizations
# ============================================================================

print(f"\n[STEP 4] Generating Visualizations")
print("-" * 80)

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────
# FIGURE 1: Coverage Over Time (7 subplots)
# ──────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 4, figsize=(24, 12))
fig.suptitle('Coverage Over Time by Dataset', fontsize=16, fontweight='bold')

for idx, ticker in enumerate(TEST_TICKERS):
    row = idx // 4
    col = idx % 4
    ax = axes[row, col]

    res = all_results[ticker]
    timesteps = range(len(res['chronos_conf']['coverages']))

    ax.plot(timesteps, res['chronos_conf']['coverages'],
            label='Chronos+Conf', alpha=0.7, linewidth=1.5, color='#3498db')
    ax.plot(timesteps, res['adapts_conf']['coverages'],
            label='AdapTS+Conf', alpha=0.7, linewidth=1.5, color='#e74c3c')
    ax.plot(timesteps, res['full_system']['coverages'],
            label='Full System', alpha=0.7, linewidth=1.5, color='#2ecc71')

    ax.axhline(y=0.9, color='black', linestyle='--', alpha=0.5, label='Target 90%')
    ax.set_xlabel('Timestep', fontsize=10)
    ax.set_ylabel('Coverage', fontsize=10)
    ax.set_title(f'{ticker}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

# Remove empty subplot
axes[1, 3].axis('off')

plt.tight_layout()
plt.savefig('results/conformal_coverage_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/conformal_coverage_analysis.png")
plt.close()

# ──────────────────────────────────────────────────────────────────────────
# FIGURE 2: Interval Width Over Time (7 subplots)
# ──────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 4, figsize=(24, 12))
fig.suptitle('Prediction Interval Width Over Time by Dataset', fontsize=16, fontweight='bold')

for idx, ticker in enumerate(TEST_TICKERS):
    row = idx // 4
    col = idx % 4
    ax = axes[row, col]

    res = all_results[ticker]
    timesteps = range(len(res['chronos_conf']['widths']))

    ax.plot(timesteps, res['chronos_conf']['widths'],
            label='Chronos+Conf', alpha=0.7, linewidth=1.5, color='#3498db')
    ax.plot(timesteps, res['adapts_conf']['widths'],
            label='AdapTS+Conf', alpha=0.7, linewidth=1.5, color='#e74c3c')
    ax.plot(timesteps, res['full_system']['widths'],
            label='Full System', alpha=0.7, linewidth=1.5, color='#2ecc71')

    ax.set_xlabel('Timestep', fontsize=10)
    ax.set_ylabel('Interval Width', fontsize=10)
    ax.set_title(f'{ticker}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

axes[1, 3].axis('off')

plt.tight_layout()
plt.savefig('results/conformal_width_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/conformal_width_analysis.png")
plt.close()

# ──────────────────────────────────────────────────────────────────────────
# FIGURE 3: MAE Over Time (7 subplots)
# ──────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 4, figsize=(24, 12))
fig.suptitle('Mean Absolute Error (MAE) Over Time by Dataset', fontsize=16, fontweight='bold')

for idx, ticker in enumerate(TEST_TICKERS):
    row = idx // 4
    col = idx % 4
    ax = axes[row, col]

    res = all_results[ticker]
    timesteps = range(len(res['chronos_conf']['mae_list']))

    ax.plot(timesteps, res['chronos_conf']['mae_list'],
            label='Chronos+Conf', alpha=0.7, linewidth=1.5, color='#3498db')
    ax.plot(timesteps, res['adapts_conf']['mae_list'],
            label='AdapTS+Conf', alpha=0.7, linewidth=1.5, color='#e74c3c')
    ax.plot(timesteps, res['full_system']['mae_list'],
            label='Full System', alpha=0.7, linewidth=1.5, color='#2ecc71')

    ax.set_xlabel('Timestep', fontsize=10)
    ax.set_ylabel('MAE', fontsize=10)
    ax.set_title(f'{ticker}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

axes[1, 3].axis('off')

plt.tight_layout()
plt.savefig('results/conformal_mae_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/conformal_mae_analysis.png")
plt.close()

# ──────────────────────────────────────────────────────────────────────────
# FIGURE 4: RMSE Over Time (7 subplots)
# ──────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 4, figsize=(24, 12))
fig.suptitle('Root Mean Squared Error (RMSE) Over Time by Dataset', fontsize=16, fontweight='bold')

for idx, ticker in enumerate(TEST_TICKERS):
    row = idx // 4
    col = idx % 4
    ax = axes[row, col]

    res = all_results[ticker]
    timesteps = range(len(res['chronos_conf']['rmse_list']))

    ax.plot(timesteps, res['chronos_conf']['rmse_list'],
            label='Chronos+Conf', alpha=0.7, linewidth=1.5, color='#3498db')
    ax.plot(timesteps, res['adapts_conf']['rmse_list'],
            label='AdapTS+Conf', alpha=0.7, linewidth=1.5, color='#e74c3c')
    ax.plot(timesteps, res['full_system']['rmse_list'],
            label='Full System', alpha=0.7, linewidth=1.5, color='#2ecc71')

    ax.set_xlabel('Timestep', fontsize=10)
    ax.set_ylabel('RMSE', fontsize=10)
    ax.set_title(f'{ticker}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

axes[1, 3].axis('off')

plt.tight_layout()
plt.savefig('results/conformal_rmse_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/conformal_rmse_analysis.png")
plt.close()

# ──────────────────────────────────────────────────────────────────────────
# FIGURE 5: Summary Bar Charts (4 metrics)
# ──────────────────────────────────────────────────────────────────────────

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Performance Summary Across All Datasets', fontsize=16, fontweight='bold')

tickers_list = list(all_results.keys())
x = np.arange(len(tickers_list))
width = 0.25

# Coverage
chronos_covs = [all_results[t]['chronos_conf']['avg_coverage']*100 for t in tickers_list]
adapts_covs = [all_results[t]['adapts_conf']['avg_coverage']*100 for t in tickers_list]
full_covs = [all_results[t]['full_system']['avg_coverage']*100 for t in tickers_list]

ax1.bar(x - width, chronos_covs, width, label='Chronos+Conf', alpha=0.8, color='#3498db')
ax1.bar(x, adapts_covs, width, label='AdapTS+Conf', alpha=0.8, color='#e74c3c')
ax1.bar(x + width, full_covs, width, label='Full System', alpha=0.8, color='#2ecc71')
ax1.axhline(y=90, color='black', linestyle='--', linewidth=2, label='Target 90%')
ax1.set_xlabel('Dataset', fontsize=12)
ax1.set_ylabel('Coverage (%)', fontsize=12)
ax1.set_title('Average Coverage', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(tickers_list, rotation=45, ha='right')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# Width
chronos_widths = [all_results[t]['chronos_conf']['avg_width'] for t in tickers_list]
adapts_widths = [all_results[t]['adapts_conf']['avg_width'] for t in tickers_list]
full_widths = [all_results[t]['full_system']['avg_width'] for t in tickers_list]

ax2.bar(x - width, chronos_widths, width, label='Chronos+Conf', alpha=0.8, color='#3498db')
ax2.bar(x, adapts_widths, width, label='AdapTS+Conf', alpha=0.8, color='#e74c3c')
ax2.bar(x + width, full_widths, width, label='Full System', alpha=0.8, color='#2ecc71')
ax2.set_xlabel('Dataset', fontsize=12)
ax2.set_ylabel('Interval Width', fontsize=12)
ax2.set_title('Average Interval Width (Lower is better)', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(tickers_list, rotation=45, ha='right')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# MAE
chronos_maes = [all_results[t]['chronos_conf']['avg_mae'] for t in tickers_list]
adapts_maes = [all_results[t]['adapts_conf']['avg_mae'] for t in tickers_list]
full_maes = [all_results[t]['full_system']['avg_mae'] for t in tickers_list]

ax3.bar(x - width, chronos_maes, width, label='Chronos+Conf', alpha=0.8, color='#3498db')
ax3.bar(x, adapts_maes, width, label='AdapTS+Conf', alpha=0.8, color='#e74c3c')
ax3.bar(x + width, full_maes, width, label='Full System', alpha=0.8, color='#2ecc71')
ax3.set_xlabel('Dataset', fontsize=12)
ax3.set_ylabel('MAE', fontsize=12)
ax3.set_title('Average Mean Absolute Error (Lower is better)', fontsize=13, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(tickers_list, rotation=45, ha='right')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# RMSE
chronos_rmses = [all_results[t]['chronos_conf']['avg_rmse'] for t in tickers_list]
adapts_rmses = [all_results[t]['adapts_conf']['avg_rmse'] for t in tickers_list]
full_rmses = [all_results[t]['full_system']['avg_rmse'] for t in tickers_list]

ax4.bar(x - width, chronos_rmses, width, label='Chronos+Conf', alpha=0.8, color='#3498db')
ax4.bar(x, adapts_rmses, width, label='AdapTS+Conf', alpha=0.8, color='#e74c3c')
ax4.bar(x + width, full_rmses, width, label='Full System', alpha=0.8, color='#2ecc71')
ax4.set_xlabel('Dataset', fontsize=12)
ax4.set_ylabel('RMSE', fontsize=12)
ax4.set_title('Average Root Mean Squared Error (Lower is better)', fontsize=13, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(tickers_list, rotation=45, ha='right')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/conformal_summary_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/conformal_summary_analysis.png")
plt.close()

# ──────────────────────────────────────────────────────────────────────────
# FIGURE 6: Ensemble Weights Over Time (for full system)
# ──────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 4, figsize=(24, 12))
fig.suptitle('Ensemble Weights Over Time (Full System)', fontsize=16, fontweight='bold')

for idx, ticker in enumerate(TEST_TICKERS):
    row = idx // 4
    col = idx % 4
    ax = axes[row, col]

    res = all_results[ticker]
    timesteps = range(len(res['full_system']['fm_weights']))

    ax.plot(timesteps, res['full_system']['fm_weights'],
            label='FM Weight', alpha=0.7, linewidth=1.5, color='#3498db')
    ax.plot(timesteps, res['full_system']['adapts_weights'],
            label='AdapTS Weight', alpha=0.7, linewidth=1.5, color='#e74c3c')

    ax.set_xlabel('Timestep', fontsize=10)
    ax.set_ylabel('Weight', fontsize=10)
    ax.set_title(f'{ticker}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

axes[1, 3].axis('off')

plt.tight_layout()
plt.savefig('results/conformal_weights_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/conformal_weights_analysis.png")
plt.close()

# ============================================================================
# STEP 5: Print Summary Table
# ============================================================================

print(f"\n{'='*80}")
print("COMPREHENSIVE RESULTS SUMMARY")
print(f"{'='*80}\n")

print(f"{'Dataset':<8} {'Method':<20} {'Coverage':<12} {'Width':<12} {'MAE':<12} {'RMSE':<12}")
print("-" * 90)

for ticker in tickers_list:
    res = all_results[ticker]
    print(f"{ticker:<8} Chronos+Conf        {res['chronos_conf']['avg_coverage']*100:>6.1f}%      {res['chronos_conf']['avg_width']:>8.4f}    {res['chronos_conf']['avg_mae']:>8.4f}    {res['chronos_conf']['avg_rmse']:>8.4f}")
    print(f"{'':8} AdapTS+Conf         {res['adapts_conf']['avg_coverage']*100:>6.1f}%      {res['adapts_conf']['avg_width']:>8.4f}    {res['adapts_conf']['avg_mae']:>8.4f}    {res['adapts_conf']['avg_rmse']:>8.4f}")
    print(f"{'':8} Full System         {res['full_system']['avg_coverage']*100:>6.1f}%      {res['full_system']['avg_width']:>8.4f}    {res['full_system']['avg_mae']:>8.4f}    {res['full_system']['avg_rmse']:>8.4f}")
    print("-" * 90)

# Overall averages
print("\nOVERALL AVERAGES ACROSS ALL DATASETS:")
print("-" * 90)

avg_chronos_cov = np.mean([all_results[t]['chronos_conf']['avg_coverage'] for t in tickers_list]) * 100
avg_adapts_cov = np.mean([all_results[t]['adapts_conf']['avg_coverage'] for t in tickers_list]) * 100
avg_full_cov = np.mean([all_results[t]['full_system']['avg_coverage'] for t in tickers_list]) * 100

avg_chronos_width = np.mean([all_results[t]['chronos_conf']['avg_width'] for t in tickers_list])
avg_adapts_width = np.mean([all_results[t]['adapts_conf']['avg_width'] for t in tickers_list])
avg_full_width = np.mean([all_results[t]['full_system']['avg_width'] for t in tickers_list])

avg_chronos_mae = np.mean([all_results[t]['chronos_conf']['avg_mae'] for t in tickers_list])
avg_adapts_mae = np.mean([all_results[t]['adapts_conf']['avg_mae'] for t in tickers_list])
avg_full_mae = np.mean([all_results[t]['full_system']['avg_mae'] for t in tickers_list])

avg_chronos_rmse = np.mean([all_results[t]['chronos_conf']['avg_rmse'] for t in tickers_list])
avg_adapts_rmse = np.mean([all_results[t]['adapts_conf']['avg_rmse'] for t in tickers_list])
avg_full_rmse = np.mean([all_results[t]['full_system']['avg_rmse'] for t in tickers_list])

print(f"Chronos+Conf:       Coverage={avg_chronos_cov:>6.1f}%  Width={avg_chronos_width:>8.4f}  MAE={avg_chronos_mae:>8.4f}  RMSE={avg_chronos_rmse:>8.4f}")
print(f"AdapTS+Conf:        Coverage={avg_adapts_cov:>6.1f}%  Width={avg_adapts_width:>8.4f}  MAE={avg_adapts_mae:>8.4f}  RMSE={avg_adapts_rmse:>8.4f}")
print(f"Full System:        Coverage={avg_full_cov:>6.1f}%  Width={avg_full_width:>8.4f}  MAE={avg_full_mae:>8.4f}  RMSE={avg_full_rmse:>8.4f}")

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETE!")
print("="*80)

print("\nGenerated files:")
print(f"  - {results_file}")
print("  - results/conformal_coverage_analysis.png")
print("  - results/conformal_width_analysis.png")
print("  - results/conformal_mae_analysis.png")
print("  - results/conformal_rmse_analysis.png")
print("  - results/conformal_summary_analysis.png")
print("  - results/conformal_weights_analysis.png")

print("\nKey Findings:")
print("  1. Coverage: How well do prediction intervals contain true values?")
print("  2. Width: Are intervals tight or too wide?")
print("  3. MAE/RMSE: Point forecast accuracy")
print("  4. Weights: How does the system balance FM vs local adaptation?")
print("\n" + "="*80)
