import numpy as np
import torch
import matplotlib.pyplot as plt
from adapts.data import TimeSeriesDataLoader
from adapts.models import BaselineFM
from adapts.models import AdapTSForecaster
from adapts.uncertainty import ConformalPredictor
from collections import deque

print("="*70)
print("Comparing: FM+Conformal vs AdapTS+Conformal vs FM+AdapTS+Conformal")
print("="*70)

# Test datasets
TEST_TICKERS = ['GOOGL', 'MSFT', 'WMT', 'COST', 'USO', 'CVX', 'UNH']

# Load trained FM
fm = BaselineFM(input_dim=1, hidden_dim=128, num_layers=2, pred_len=24)
fm.load_state_dict(torch.load('trained_fm_focused.pth'))

# Store results for all datasets
all_results = {}

for ticker in TEST_TICKERS:
    print(f"\n{'='*70}")
    print(f"Processing {ticker}...")
    print(f"{'='*70}")
    
    # Load data
    loader = TimeSeriesDataLoader(source='stock')
    train_data, test_data = loader.load(ticker=ticker, start='2015-01-01', end='2024-01-01')
    
    # Limit test data for speed
    test_data = test_data[:400]
    
    seq_len = 96
    pred_len = 24
    
    # ========================================================================
    # METHOD 1: FM + Conformal (Baseline)
    # ========================================================================
    print("\n[1] FM + Conformal...")
    
    fm_conf = ConformalPredictor(alpha=0.1, calibration_size=200)
    fm_coverages = []
    fm_widths = []
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
        covered = sum(1 for j in range(len(y_true)) 
                     if result['lower'][j] <= y_true[j] <= result['upper'][j])
        coverage = covered / len(y_true)
        width = np.mean(result['upper'] - result['lower'])
        
        fm_coverages.append(coverage)
        fm_widths.append(width)
        fm_predictions.append(result)
    
    print(f"   Avg Coverage: {np.mean(fm_coverages)*100:.1f}%")
    print(f"   Avg Width: {np.mean(fm_widths):.4f}")
    
    # ========================================================================
    # METHOD 2: AdapTS + Conformal (No FM in final prediction)
    # ========================================================================
    print("\n[2] AdapTS-Forecaster + Conformal (no FM)...")
    
    adapts_forecaster = AdapTSForecaster(alpha=0.1, window_size=5, pred_len=24)
    adapts_conf = ConformalPredictor(alpha=0.1, calibration_size=200)
    adapts_coverages = []
    adapts_widths = []
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
        covered = sum(1 for j in range(len(y_true)) 
                     if result['lower'][j] <= y_true[j] <= result['upper'][j])
        coverage = covered / len(y_true)
        width = np.mean(result['upper'] - result['lower'])
        
        adapts_coverages.append(coverage)
        adapts_widths.append(width)
        adapts_predictions.append(result)
    
    print(f"   Avg Coverage: {np.mean(adapts_coverages)*100:.1f}%")
    print(f"   Avg Width: {np.mean(adapts_widths):.4f}")
    
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
    full_predictions = []
    
    for i in range(len(test_data) - seq_len - pred_len):
        x = test_data[i:i+seq_len]
        y_true = test_data[i+seq_len:i+seq_len+pred_len]
        
        # Full AdapTS prediction
        result = full_adapts.predict(x, return_components=False)
        
        # Update
        full_adapts.update(x, y_true)
        
        # Calculate metrics
        covered = sum(1 for j in range(len(y_true)) 
                     if result['lower'][j] <= y_true[j] <= result['upper'][j])
        coverage = covered / len(y_true)
        width = np.mean(result['upper'] - result['lower'])
        
        full_coverages.append(coverage)
        full_widths.append(width)
        full_predictions.append(result)
    
    print(f"   Avg Coverage: {np.mean(full_coverages)*100:.1f}%")
    print(f"   Avg Width: {np.mean(full_widths):.4f}")
    
    # Store results
    all_results[ticker] = {
        'fm_conf': {
            'coverages': fm_coverages,
            'widths': fm_widths,
            'avg_coverage': np.mean(fm_coverages),
            'avg_width': np.mean(fm_widths)
        },
        'adapts_conf': {
            'coverages': adapts_coverages,
            'widths': adapts_widths,
            'avg_coverage': np.mean(adapts_coverages),
            'avg_width': np.mean(adapts_widths)
        },
        'full_adapts': {
            'coverages': full_coverages,
            'widths': full_widths,
            'avg_coverage': np.mean(full_coverages),
            'avg_width': np.mean(full_widths)
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

# Plot 3: Summary bar charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

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

plt.tight_layout()
plt.savefig('summary_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: summary_comparison.png")

# ============================================================================
# SUMMARY TABLE
# ============================================================================

print(f"\n{'='*70}")
print("SUMMARY RESULTS")
print(f"{'='*70}\n")

print(f"{'Dataset':<8} {'Method':<20} {'Avg Coverage':<15} {'Avg Width':<12}")
print("-" * 70)

for ticker in tickers_list:
    res = all_results[ticker]
    print(f"{ticker:<8} FM+Conf            {res['fm_conf']['avg_coverage']*100:>6.1f}%         {res['fm_conf']['avg_width']:>8.4f}")
    print(f"{'':8} AdapTS+Conf         {res['adapts_conf']['avg_coverage']*100:>6.1f}%         {res['adapts_conf']['avg_width']:>8.4f}")
    print(f"{'':8} FM+AdapTS+Conf      {res['full_adapts']['avg_coverage']*100:>6.1f}%         {res['full_adapts']['avg_width']:>8.4f}")
    print("-" * 70)

print("\n✓ Analysis complete!")
print("\nGenerated files:")
print("  - coverage_comparison_all_datasets.png")
print("  - width_comparison_all_datasets.png")
print("  - summary_comparison.png")