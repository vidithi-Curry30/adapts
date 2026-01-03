"""
Compare Chronos FM (Amazon's pre-trained FM) with and without AdapTS
Full comparison across 7 stock datasets with visualization
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
from adapts.metrics import TimeSeriesMetrics

print("="*70)
print("Comparing: Chronos+Conformal vs Chronos+AdapTS+Conformal")
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

TEST_TICKERS = ['GOOGL', 'MSFT', 'WMT', 'COST', 'USO', 'CVX', 'UNH']
all_results = {}

for ticker in TEST_TICKERS:
    print(f"\n{'='*70}")
    print(f"Processing {ticker}...")
    print(f"{'='*70}")

    # Load data
    loader = TimeSeriesDataLoader(source='stock')
    train_data, test_data = loader.load(ticker=ticker, start='2015-01-01', end='2024-01-01')

    seq_len = 96
    pred_len = 24

    # ========================================================================
    # METHOD 1: Chronos + Conformal (Zero-shot)
    # ========================================================================
    print("\n[1] Chronos + Conformal (zero-shot)...")

    chronos_conf = ConformalPredictor(alpha=0.1, calibration_size=200)
    chronos_coverages = []
    chronos_widths = []
    chronos_mae_list = []
    chronos_rmse_list = []

    n_steps = len(test_data) - seq_len - pred_len
    for i in range(n_steps):
        x = test_data[i:i+seq_len]
        y_true = test_data[i+seq_len:i+seq_len+pred_len]

        # Chronos prediction
        pred = chronos_fm.predict(x).flatten()

        # Conformal interval
        result = chronos_conf.predict_with_interval(pred)
        chronos_conf.update_residuals(y_true, pred)

        # Metrics
        chronos_coverages.append(TimeSeriesMetrics.coverage(y_true, result['lower'], result['upper']))
        chronos_widths.append(TimeSeriesMetrics.interval_width(result['lower'], result['upper']))
        chronos_mae_list.append(TimeSeriesMetrics.mae(y_true, pred))
        chronos_rmse_list.append(TimeSeriesMetrics.rmse(y_true, pred))

    print(f"   Avg Coverage: {np.mean(chronos_coverages)*100:.1f}%")
    print(f"   Avg Width: {np.mean(chronos_widths):.4f}")
    print(f"   Avg MAE: {np.mean(chronos_mae_list):.4f}")
    print(f"   Avg RMSE: {np.mean(chronos_rmse_list):.4f}")

    # ========================================================================
    # METHOD 2: Chronos + AdapTS + Conformal
    # ========================================================================
    print("\n[2] Chronos + AdapTS + Conformal...")

    forecaster = AdapTSForecaster(alpha=0.1, window_size=5, pred_len=24)
    weighter = AdapTSWeighter(memory_size=20, beta=0.9)
    adapts_conf = ConformalPredictor(alpha=0.1, calibration_size=200)

    adapts_coverages = []
    adapts_widths = []
    adapts_mae_list = []
    adapts_rmse_list = []

    for i in range(n_steps):
        x = test_data[i:i+seq_len]
        y_true = test_data[i+seq_len:i+seq_len+pred_len]

        # Both predictions
        chronos_pred = chronos_fm.predict(x).flatten()
        local_pred = forecaster.predict(x)
        combined_pred = weighter.combine_predictions(chronos_pred, local_pred)

        # Conformal interval
        result = adapts_conf.predict_with_interval(combined_pred)

        # Update
        forecaster.update(x, y_true)
        weighter.update_residuals(y_true, chronos_pred, local_pred)
        adapts_conf.update_residuals(y_true, combined_pred)

        # Metrics
        adapts_coverages.append(TimeSeriesMetrics.coverage(y_true, result['lower'], result['upper']))
        adapts_widths.append(TimeSeriesMetrics.interval_width(result['lower'], result['upper']))
        adapts_mae_list.append(TimeSeriesMetrics.mae(y_true, combined_pred))
        adapts_rmse_list.append(TimeSeriesMetrics.rmse(y_true, combined_pred))

    print(f"   Avg Coverage: {np.mean(adapts_coverages)*100:.1f}%")
    print(f"   Avg Width: {np.mean(adapts_widths):.4f}")
    print(f"   Avg MAE: {np.mean(adapts_mae_list):.4f}")
    print(f"   Avg RMSE: {np.mean(adapts_rmse_list):.4f}")

    mae_improvement = (np.mean(chronos_mae_list) - np.mean(adapts_mae_list)) / np.mean(chronos_mae_list) * 100
    rmse_improvement = (np.mean(chronos_rmse_list) - np.mean(adapts_rmse_list)) / np.mean(chronos_rmse_list) * 100
    print(f"   ✓ MAE Improvement: {mae_improvement:.1f}%")
    print(f"   ✓ RMSE Improvement: {rmse_improvement:.1f}%")

    # Store results
    all_results[ticker] = {
        'chronos_conf': {
            'coverages': chronos_coverages,
            'widths': chronos_widths,
            'mae_list': chronos_mae_list,
            'rmse_list': chronos_rmse_list,
            'avg_coverage': np.mean(chronos_coverages),
            'avg_width': np.mean(chronos_widths),
            'avg_mae': np.mean(chronos_mae_list),
            'avg_rmse': np.mean(chronos_rmse_list)
        },
        'chronos_adapts': {
            'coverages': adapts_coverages,
            'widths': adapts_widths,
            'mae_list': adapts_mae_list,
            'rmse_list': adapts_rmse_list,
            'avg_coverage': np.mean(adapts_coverages),
            'avg_width': np.mean(adapts_widths),
            'avg_mae': np.mean(adapts_mae_list),
            'avg_rmse': np.mean(adapts_rmse_list)
        }
    }

# ============================================================================
# VISUALIZATION: Coverage and Width Plots
# ============================================================================

print(f"\n{'='*70}")
print("Generating comparison plots...")
print(f"{'='*70}")

# Plot 1: Coverage over time
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for idx, ticker in enumerate(TEST_TICKERS):
    row = idx // 4
    col = idx % 4

    ax = axes[row, col]

    res = all_results[ticker]

    # Coverage over time
    timesteps = range(len(res['chronos_conf']['coverages']))

    ax.plot(timesteps, res['chronos_conf']['coverages'],
            label='Chronos+Conf', alpha=0.7, linewidth=1.5, color='skyblue')
    ax.plot(timesteps, res['chronos_adapts']['coverages'],
            label='Chronos+AdapTS+Conf', alpha=0.7, linewidth=1.5, color='lightgreen')

    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='Target 90%')
    ax.set_xlabel('Timestep', fontsize=9)
    ax.set_ylabel('Coverage', fontsize=9)
    ax.set_title(f'{ticker} - Coverage Over Time', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('chronos_coverage_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: chronos_coverage_comparison.png")

# Plot 2: Width over time
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for idx, ticker in enumerate(TEST_TICKERS):
    row = idx // 4
    col = idx % 4

    ax = axes[row, col]

    res = all_results[ticker]
    timesteps = range(len(res['chronos_conf']['widths']))

    ax.plot(timesteps, res['chronos_conf']['widths'],
            label='Chronos+Conf', alpha=0.7, linewidth=1.5, color='skyblue')
    ax.plot(timesteps, res['chronos_adapts']['widths'],
            label='Chronos+AdapTS+Conf', alpha=0.7, linewidth=1.5, color='lightgreen')

    ax.set_xlabel('Timestep', fontsize=9)
    ax.set_ylabel('Interval Width', fontsize=9)
    ax.set_title(f'{ticker} - Interval Width Over Time', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chronos_width_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: chronos_width_comparison.png")

# Plot 3: MAE over time
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for idx, ticker in enumerate(TEST_TICKERS):
    row = idx // 4
    col = idx % 4

    ax = axes[row, col]

    res = all_results[ticker]
    timesteps = range(len(res['chronos_conf']['mae_list']))

    ax.plot(timesteps, res['chronos_conf']['mae_list'],
            label='Chronos+Conf', alpha=0.7, linewidth=1.5, color='skyblue')
    ax.plot(timesteps, res['chronos_adapts']['mae_list'],
            label='Chronos+AdapTS+Conf', alpha=0.7, linewidth=1.5, color='lightgreen')

    ax.set_xlabel('Timestep', fontsize=9)
    ax.set_ylabel('MAE', fontsize=9)
    ax.set_title(f'{ticker} - MAE Over Time', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chronos_mae_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: chronos_mae_comparison.png")

# Plot 4: RMSE over time
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for idx, ticker in enumerate(TEST_TICKERS):
    row = idx // 4
    col = idx % 4

    ax = axes[row, col]

    res = all_results[ticker]
    timesteps = range(len(res['chronos_conf']['rmse_list']))

    ax.plot(timesteps, res['chronos_conf']['rmse_list'],
            label='Chronos+Conf', alpha=0.7, linewidth=1.5, color='skyblue')
    ax.plot(timesteps, res['chronos_adapts']['rmse_list'],
            label='Chronos+AdapTS+Conf', alpha=0.7, linewidth=1.5, color='lightgreen')

    ax.set_xlabel('Timestep', fontsize=9)
    ax.set_ylabel('RMSE', fontsize=9)
    ax.set_title(f'{ticker} - RMSE Over Time', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chronos_rmse_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: chronos_rmse_comparison.png")

# Plot 5: Summary bar charts with MAE and RMSE
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Coverage summary
tickers_list = list(all_results.keys())
chronos_covs = [all_results[t]['chronos_conf']['avg_coverage']*100 for t in tickers_list]
adapts_covs = [all_results[t]['chronos_adapts']['avg_coverage']*100 for t in tickers_list]

x = np.arange(len(tickers_list))
width = 0.35

ax1.bar(x - width/2, chronos_covs, width, label='Chronos+Conf', alpha=0.8, color='skyblue')
ax1.bar(x + width/2, adapts_covs, width, label='Chronos+AdapTS+Conf', alpha=0.8, color='lightgreen')

ax1.axhline(y=90, color='red', linestyle='--', linewidth=2, label='Target 90%')
ax1.set_xlabel('Dataset', fontsize=11)
ax1.set_ylabel('Average Coverage (%)', fontsize=11)
ax1.set_title('Average Coverage by Method', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(tickers_list, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Width summary
chronos_widths = [all_results[t]['chronos_conf']['avg_width'] for t in tickers_list]
adapts_widths = [all_results[t]['chronos_adapts']['avg_width'] for t in tickers_list]

ax2.bar(x - width/2, chronos_widths, width, label='Chronos+Conf', alpha=0.8, color='skyblue')
ax2.bar(x + width/2, adapts_widths, width, label='Chronos+AdapTS+Conf', alpha=0.8, color='lightgreen')

ax2.set_xlabel('Dataset', fontsize=11)
ax2.set_ylabel('Average Interval Width', fontsize=11)
ax2.set_title('Average Interval Width by Method', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(tickers_list, rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# MAE summary
chronos_maes = [all_results[t]['chronos_conf']['avg_mae'] for t in tickers_list]
adapts_maes = [all_results[t]['chronos_adapts']['avg_mae'] for t in tickers_list]

ax3.bar(x - width/2, chronos_maes, width, label='Chronos+Conf', alpha=0.8, color='skyblue')
ax3.bar(x + width/2, adapts_maes, width, label='Chronos+AdapTS+Conf', alpha=0.8, color='lightgreen')

ax3.set_xlabel('Dataset', fontsize=11)
ax3.set_ylabel('Average MAE', fontsize=11)
ax3.set_title('Average MAE by Method', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(tickers_list, rotation=45, ha='right')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# RMSE summary
chronos_rmses = [all_results[t]['chronos_conf']['avg_rmse'] for t in tickers_list]
adapts_rmses = [all_results[t]['chronos_adapts']['avg_rmse'] for t in tickers_list]

ax4.bar(x - width/2, chronos_rmses, width, label='Chronos+Conf', alpha=0.8, color='skyblue')
ax4.bar(x + width/2, adapts_rmses, width, label='Chronos+AdapTS+Conf', alpha=0.8, color='lightgreen')

ax4.set_xlabel('Dataset', fontsize=11)
ax4.set_ylabel('Average RMSE', fontsize=11)
ax4.set_title('Average RMSE by Method', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(tickers_list, rotation=45, ha='right')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('chronos_summary_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: chronos_summary_comparison.png")

# ============================================================================
# SUMMARY TABLE
# ============================================================================

print(f"\n{'='*70}")
print("SUMMARY RESULTS")
print(f"{'='*70}\n")

print(f"{'Dataset':<8} {'Method':<25} {'Avg Coverage':<15} {'Avg Width':<12} {'Avg MAE':<12} {'Avg RMSE':<12}")
print("-" * 100)

for ticker in tickers_list:
    res = all_results[ticker]
    print(f"{ticker:<8} Chronos+Conf          {res['chronos_conf']['avg_coverage']*100:>6.1f}%         {res['chronos_conf']['avg_width']:>8.4f}     {res['chronos_conf']['avg_mae']:>8.4f}     {res['chronos_conf']['avg_rmse']:>8.4f}")
    print(f"{'':8} Chronos+AdapTS+Conf    {res['chronos_adapts']['avg_coverage']*100:>6.1f}%         {res['chronos_adapts']['avg_width']:>8.4f}     {res['chronos_adapts']['avg_mae']:>8.4f}     {res['chronos_adapts']['avg_rmse']:>8.4f}")
    print("-" * 100)

print("\n✓ Analysis complete!")
print("\nGenerated files:")
print("  - chronos_coverage_comparison.png")
print("  - chronos_width_comparison.png")
print("  - chronos_mae_comparison.png")
print("  - chronos_rmse_comparison.png")
print("  - chronos_summary_comparison.png")
