import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from adapts.data import TimeSeriesDataLoader
from adapts.models import BaselineFM
from adapts.models import AdapTSForecaster
from adapts.uncertainty import ConformalPredictor
from adapts.metrics import TimeSeriesMetrics
from collections import deque

print("="*70)
print("Comparing: FM+Conformal vs AdapTS+Conformal vs FM+AdapTS+Conformal")
print("="*70)

# Test datasets
TEST_TICKERS = ['GOOGL', 'MSFT', 'WMT', 'COST', 'USO', 'CVX', 'UNH']

# Load trained FM
# Get the script's directory to construct absolute path
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'trained_fm_focused.pth')

fm = BaselineFM(input_dim=1, hidden_dim=128, num_layers=2, pred_len=24)
fm.load_state_dict(torch.load(model_path))

# Store results for all datasets
all_results = {}

for ticker in TEST_TICKERS:
    print(f"\n{'='*70}")
    print(f"Processing {ticker}...")
    print(f"{'='*70}")
    
    # Load data (use all data for evaluation with split_ratio=0.0)
    loader = TimeSeriesDataLoader(source='stock', split_ratio=0.0)
    train_data, test_data = loader.load(ticker=ticker, start='2015-01-01', end='2024-01-01')
    
    seq_len = 96
    pred_len = 24
    
    # ========================================================================
    # METHOD 1: FM + Conformal (Baseline)
    # ========================================================================
    print("\n[1] FM + Conformal...")
    
    fm_conf = ConformalPredictor(alpha=0.1, calibration_size=200)
    fm_coverages = []
    fm_widths = []
    fm_mae_list = []
    fm_rmse_list = []
    fm_predictions = []
    
    for i in range(len(test_data) - seq_len - pred_len):
        x = test_data[i:i+seq_len]
        y_true = test_data[i+seq_len:i+seq_len+pred_len]
        
        # FM prediction
        pred = fm.predict(x.reshape(1, -1)).flatten()
        
        # Add conformal interval
        result = fm_conf.predict_with_interval(pred)
        
        # Update conformal
        fm_conf.update_residuals(y_true, pred)
        
        # Calculate metrics
        fm_coverages.append(TimeSeriesMetrics.coverage(y_true, result['lower'], result['upper']))
        fm_widths.append(TimeSeriesMetrics.interval_width(result['lower'], result['upper']))
        fm_mae_list.append(TimeSeriesMetrics.mae(y_true, pred))
        fm_rmse_list.append(TimeSeriesMetrics.rmse(y_true, pred))
        fm_predictions.append(result)
    
    print(f"   Avg Coverage: {np.mean(fm_coverages)*100:.1f}%")
    print(f"   Avg Width: {np.mean(fm_widths):.4f}")
    print(f"   Avg MAE: {np.mean(fm_mae_list):.4f}")
    print(f"   Avg RMSE: {np.mean(fm_rmse_list):.4f}")
    
    # ========================================================================
    # METHOD 2: AdapTS + Conformal (No FM in final prediction)
    # ========================================================================
    print("\n[2] AdapTS-Forecaster + Conformal (no FM)...")
    
    adapts_forecaster = AdapTSForecaster(alpha=0.1, window_size=5, pred_len=24)
    adapts_conf = ConformalPredictor(alpha=0.1, calibration_size=200)
    adapts_coverages = []
    adapts_widths = []
    adapts_mae_list = []
    adapts_rmse_list = []
    adapts_predictions = []
    
    for i in range(len(test_data) - seq_len - pred_len):
        x = test_data[i:i+seq_len]
        y_true = test_data[i+seq_len:i+seq_len+pred_len]
        
        # AdapTS-Forecaster prediction only
        pred = adapts_forecaster.predict(x)
        
        # Add conformal interval
        result = adapts_conf.predict_with_interval(pred)
        
        # Update both
        adapts_forecaster.update(x, y_true)
        adapts_conf.update_residuals(y_true, pred)
        
        # Calculate metrics
        adapts_coverages.append(TimeSeriesMetrics.coverage(y_true, result['lower'], result['upper']))
        adapts_widths.append(TimeSeriesMetrics.interval_width(result['lower'], result['upper']))
        adapts_mae_list.append(TimeSeriesMetrics.mae(y_true, pred))
        adapts_rmse_list.append(TimeSeriesMetrics.rmse(y_true, pred))
        adapts_predictions.append(result)
    
    print(f"   Avg Coverage: {np.mean(adapts_coverages)*100:.1f}%")
    print(f"   Avg Width: {np.mean(adapts_widths):.4f}")
    print(f"   Avg MAE: {np.mean(adapts_mae_list):.4f}")
    print(f"   Avg RMSE: {np.mean(adapts_rmse_list):.4f}")
    
    # ========================================================================
    # METHOD 3: FM + AdapTS + Conformal (Full System)
    # ========================================================================
    print("\n[3] FM + AdapTS-Forecaster + Weighter + Conformal (Full)...")
    
    from adapts import AdapTS
    
    fm_fresh = BaselineFM(input_dim=1, hidden_dim=128, num_layers=2, pred_len=24)
    fm_fresh.load_state_dict(fm.state_dict())
    
    full_adapts = AdapTS(
        foundation_model=fm_fresh,
        seq_len=seq_len,
        pred_len=pred_len,
        window_size=5,
        alpha_ridge=0.1,
        beta=0.9,
        conf_alpha=0.1,
        calibration_size=200
    )
    
    full_coverages = []
    full_widths = []
    full_mae_list = []
    full_rmse_list = []
    full_predictions = []
    
    for i in range(len(test_data) - seq_len - pred_len):
        x = test_data[i:i+seq_len]
        y_true = test_data[i+seq_len:i+seq_len+pred_len]
        
        # Full AdapTS prediction
        result = full_adapts.predict(x, return_components=False)
        
        # Update
        full_adapts.update(x, y_true)
        
        # Calculate metrics
        full_coverages.append(TimeSeriesMetrics.coverage(y_true, result['lower'], result['upper']))
        full_widths.append(TimeSeriesMetrics.interval_width(result['lower'], result['upper']))
        full_mae_list.append(TimeSeriesMetrics.mae(y_true, result['prediction']))
        full_rmse_list.append(TimeSeriesMetrics.rmse(y_true, result['prediction']))
        full_predictions.append(result)
    
    print(f"   Avg Coverage: {np.mean(full_coverages)*100:.1f}%")
    print(f"   Avg Width: {np.mean(full_widths):.4f}")
    print(f"   Avg MAE: {np.mean(full_mae_list):.4f}")
    print(f"   Avg RMSE: {np.mean(full_rmse_list):.4f}")
    
    # Store results
    all_results[ticker] = {
        'fm_conf': {
            'coverages': fm_coverages,
            'widths': fm_widths,
            'mae_list': fm_mae_list,
            'rmse_list': fm_rmse_list,
            'avg_coverage': np.mean(fm_coverages),
            'avg_width': np.mean(fm_widths),
            'avg_mae': np.mean(fm_mae_list),
            'avg_rmse': np.mean(fm_rmse_list)
        },
        'adapts_conf': {
            'coverages': adapts_coverages,
            'widths': adapts_widths,
            'mae_list': adapts_mae_list,
            'rmse_list': adapts_rmse_list,
            'avg_coverage': np.mean(adapts_coverages),
            'avg_width': np.mean(adapts_widths),
            'avg_mae': np.mean(adapts_mae_list),
            'avg_rmse': np.mean(adapts_rmse_list)
        },
        'full_adapts': {
            'coverages': full_coverages,
            'widths': full_widths,
            'mae_list': full_mae_list,
            'rmse_list': full_rmse_list,
            'avg_coverage': np.mean(full_coverages),
            'avg_width': np.mean(full_widths),
            'avg_mae': np.mean(full_mae_list),
            'avg_rmse': np.mean(full_rmse_list)
        }
    }

# ============================================================================
# VISUALIZATION: Coverage and Width Plots
# ============================================================================

print(f"\n{'='*70}")
print("Generating comparison plots...")
print(f"{'='*70}")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for idx, ticker in enumerate(TEST_TICKERS):
    row = idx // 4
    col = idx % 4
    
    ax = axes[row, col]
    
    res = all_results[ticker]
    
    # Plot 1: Coverage over time
    timesteps = range(len(res['fm_conf']['coverages']))
    
    ax.plot(timesteps, res['fm_conf']['coverages'], 
            label='FM+Conf', alpha=0.7, linewidth=1.5)
    ax.plot(timesteps, res['adapts_conf']['coverages'], 
            label='AdapTS+Conf', alpha=0.7, linewidth=1.5)
    ax.plot(timesteps, res['full_adapts']['coverages'], 
            label='FM+AdapTS+Conf', alpha=0.7, linewidth=1.5)
    
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='Target 90%')
    ax.set_xlabel('Timestep', fontsize=9)
    ax.set_ylabel('Coverage', fontsize=9)
    ax.set_title(f'{ticker} - Coverage Over Time', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('coverage_comparison_all_datasets.png', dpi=300, bbox_inches='tight')
print("✓ Saved: coverage_comparison_all_datasets.png")

# Plot 2: Width comparison
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for idx, ticker in enumerate(TEST_TICKERS):
    row = idx // 4
    col = idx % 4
    
    ax = axes[row, col]
    
    res = all_results[ticker]
    timesteps = range(len(res['fm_conf']['widths']))
    
    ax.plot(timesteps, res['fm_conf']['widths'], 
            label='FM+Conf', alpha=0.7, linewidth=1.5)
    ax.plot(timesteps, res['adapts_conf']['widths'], 
            label='AdapTS+Conf', alpha=0.7, linewidth=1.5)
    ax.plot(timesteps, res['full_adapts']['widths'], 
            label='FM+AdapTS+Conf', alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Timestep', fontsize=9)
    ax.set_ylabel('Interval Width', fontsize=9)
    ax.set_title(f'{ticker} - Interval Width Over Time', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('width_comparison_all_datasets.png', dpi=300, bbox_inches='tight')
print("✓ Saved: width_comparison_all_datasets.png")

# Plot 3: MAE over time
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for idx, ticker in enumerate(TEST_TICKERS):
    row = idx // 4
    col = idx % 4

    ax = axes[row, col]

    res = all_results[ticker]
    timesteps = range(len(res['fm_conf']['mae_list']))

    ax.plot(timesteps, res['fm_conf']['mae_list'],
            label='FM+Conf', alpha=0.7, linewidth=1.5)
    ax.plot(timesteps, res['adapts_conf']['mae_list'],
            label='AdapTS+Conf', alpha=0.7, linewidth=1.5)
    ax.plot(timesteps, res['full_adapts']['mae_list'],
            label='FM+AdapTS+Conf', alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Timestep', fontsize=9)
    ax.set_ylabel('MAE', fontsize=9)
    ax.set_title(f'{ticker} - MAE Over Time', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mae_comparison_all_datasets.png', dpi=300, bbox_inches='tight')
print("✓ Saved: mae_comparison_all_datasets.png")

# Plot 4: RMSE over time
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for idx, ticker in enumerate(TEST_TICKERS):
    row = idx // 4
    col = idx % 4

    ax = axes[row, col]

    res = all_results[ticker]
    timesteps = range(len(res['fm_conf']['rmse_list']))

    ax.plot(timesteps, res['fm_conf']['rmse_list'],
            label='FM+Conf', alpha=0.7, linewidth=1.5)
    ax.plot(timesteps, res['adapts_conf']['rmse_list'],
            label='AdapTS+Conf', alpha=0.7, linewidth=1.5)
    ax.plot(timesteps, res['full_adapts']['rmse_list'],
            label='FM+AdapTS+Conf', alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Timestep', fontsize=9)
    ax.set_ylabel('RMSE', fontsize=9)
    ax.set_title(f'{ticker} - RMSE Over Time', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rmse_comparison_all_datasets.png', dpi=300, bbox_inches='tight')
print("✓ Saved: rmse_comparison_all_datasets.png")

# Plot 5: Summary bar charts with MAE and RMSE
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Coverage summary
tickers_list = list(all_results.keys())
fm_covs = [all_results[t]['fm_conf']['avg_coverage']*100 for t in tickers_list]
adapts_covs = [all_results[t]['adapts_conf']['avg_coverage']*100 for t in tickers_list]
full_covs = [all_results[t]['full_adapts']['avg_coverage']*100 for t in tickers_list]

x = np.arange(len(tickers_list))
width = 0.25

ax1.bar(x - width, fm_covs, width, label='FM+Conf', alpha=0.8, color='skyblue')
ax1.bar(x, adapts_covs, width, label='AdapTS+Conf', alpha=0.8, color='lightcoral')
ax1.bar(x + width, full_covs, width, label='FM+AdapTS+Conf', alpha=0.8, color='lightgreen')

ax1.axhline(y=90, color='red', linestyle='--', linewidth=2, label='Target 90%')
ax1.set_xlabel('Dataset', fontsize=11)
ax1.set_ylabel('Average Coverage (%)', fontsize=11)
ax1.set_title('Average Coverage by Method', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(tickers_list, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Width summary
fm_widths = [all_results[t]['fm_conf']['avg_width'] for t in tickers_list]
adapts_widths = [all_results[t]['adapts_conf']['avg_width'] for t in tickers_list]
full_widths = [all_results[t]['full_adapts']['avg_width'] for t in tickers_list]

ax2.bar(x - width, fm_widths, width, label='FM+Conf', alpha=0.8, color='skyblue')
ax2.bar(x, adapts_widths, width, label='AdapTS+Conf', alpha=0.8, color='lightcoral')
ax2.bar(x + width, full_widths, width, label='FM+AdapTS+Conf', alpha=0.8, color='lightgreen')

ax2.set_xlabel('Dataset', fontsize=11)
ax2.set_ylabel('Average Interval Width', fontsize=11)
ax2.set_title('Average Interval Width by Method', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(tickers_list, rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# MAE summary
fm_maes = [all_results[t]['fm_conf']['avg_mae'] for t in tickers_list]
adapts_maes = [all_results[t]['adapts_conf']['avg_mae'] for t in tickers_list]
full_maes = [all_results[t]['full_adapts']['avg_mae'] for t in tickers_list]

ax3.bar(x - width, fm_maes, width, label='FM+Conf', alpha=0.8, color='skyblue')
ax3.bar(x, adapts_maes, width, label='AdapTS+Conf', alpha=0.8, color='lightcoral')
ax3.bar(x + width, full_maes, width, label='FM+AdapTS+Conf', alpha=0.8, color='lightgreen')

ax3.set_xlabel('Dataset', fontsize=11)
ax3.set_ylabel('Average MAE', fontsize=11)
ax3.set_title('Average MAE by Method', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(tickers_list, rotation=45, ha='right')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# RMSE summary
fm_rmses = [all_results[t]['fm_conf']['avg_rmse'] for t in tickers_list]
adapts_rmses = [all_results[t]['adapts_conf']['avg_rmse'] for t in tickers_list]
full_rmses = [all_results[t]['full_adapts']['avg_rmse'] for t in tickers_list]

ax4.bar(x - width, fm_rmses, width, label='FM+Conf', alpha=0.8, color='skyblue')
ax4.bar(x, adapts_rmses, width, label='AdapTS+Conf', alpha=0.8, color='lightcoral')
ax4.bar(x + width, full_rmses, width, label='FM+AdapTS+Conf', alpha=0.8, color='lightgreen')

ax4.set_xlabel('Dataset', fontsize=11)
ax4.set_ylabel('Average RMSE', fontsize=11)
ax4.set_title('Average RMSE by Method', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(tickers_list, rotation=45, ha='right')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('summary_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: summary_comparison.png")

# ============================================================================
# SUMMARY TABLE
# ============================================================================

print(f"\n{'='*70}")
print("SUMMARY RESULTS")
print(f"{'='*70}\n")

print(f"{'Dataset':<8} {'Method':<20} {'Avg Coverage':<15} {'Avg Width':<12} {'Avg MAE':<12} {'Avg RMSE':<12}")
print("-" * 95)

for ticker in tickers_list:
    res = all_results[ticker]
    print(f"{ticker:<8} FM+Conf            {res['fm_conf']['avg_coverage']*100:>6.1f}%         {res['fm_conf']['avg_width']:>8.4f}     {res['fm_conf']['avg_mae']:>8.4f}     {res['fm_conf']['avg_rmse']:>8.4f}")
    print(f"{'':8} AdapTS+Conf         {res['adapts_conf']['avg_coverage']*100:>6.1f}%         {res['adapts_conf']['avg_width']:>8.4f}     {res['adapts_conf']['avg_mae']:>8.4f}     {res['adapts_conf']['avg_rmse']:>8.4f}")
    print(f"{'':8} FM+AdapTS+Conf      {res['full_adapts']['avg_coverage']*100:>6.1f}%         {res['full_adapts']['avg_width']:>8.4f}     {res['full_adapts']['avg_mae']:>8.4f}     {res['full_adapts']['avg_rmse']:>8.4f}")
    print("-" * 95)

print("\n✓ Analysis complete!")
print("\nGenerated files:")
print("  - coverage_comparison_all_datasets.png")
print("  - width_comparison_all_datasets.png")
print("  - mae_comparison_all_datasets.png")
print("  - rmse_comparison_all_datasets.png")
print("  - summary_comparison.png")