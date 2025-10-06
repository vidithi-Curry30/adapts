import numpy as np
import matplotlib.pyplot as plt
from adapts.data import TimeSeriesDataLoader
from adapts.models import BaselineFM
from adapts import AdapTS

print("="*70)
print("AdapTS Experiments: Hyperparameter Tuning & Comparison")
print("="*70)

# Load data
loader = TimeSeriesDataLoader(source='stock')
train_data, test_data = loader.load(ticker='AAPL', start='2020-01-01', end='2024-01-01')

fm = BaselineFM(hidden_dim=128, pred_len=24)

# Experiment 1: Compare AdapTS vs Baseline Conformal
print("\n" + "="*70)
print("EXPERIMENT 1: AdapTS vs Baseline Conformal Prediction")
print("="*70)

# Baseline: Just FM + Conformal (no adaptation)
print("\n[Baseline] Running FM + Conformal only (no AdapTS)...")
from adapts.uncertainty import ConformalPredictor

baseline_errors = []
baseline_coverage = 0
conf_baseline = ConformalPredictor(alpha=0.1, calibration_size=50)

for i in range(len(test_data) - 120):
    x = test_data[i:i+96]
    y_true = test_data[i+96:i+120]
    
    # FM prediction only
    fm_pred = fm.predict(x.reshape(1, -1)).flatten()
    
    # Conformal interval
    result = conf_baseline.predict_with_interval(fm_pred)
    
    # Update
    conf_baseline.update_residuals(y_true, fm_pred)
    
    # Metrics
    baseline_errors.append(np.mean(np.abs(y_true - fm_pred)))
    if np.all((y_true >= result['lower']) & (y_true <= result['upper'])):
        baseline_coverage += 1

baseline_mae = np.mean(baseline_errors)
baseline_coverage = baseline_coverage / len(baseline_errors)

print(f"\nBaseline Results:")
print(f"  MAE: {baseline_mae:.4f}")
print(f"  Coverage: {baseline_coverage*100:.1f}%")

# AdapTS
print("\n[AdapTS] Running full AdapTS system...")
adapts = AdapTS(foundation_model=fm, seq_len=96, pred_len=24)
results_adapts = adapts.evaluate_online(test_data[:len(baseline_errors)*120 + 120], verbose=False)

print(f"\nAdapTS Results:")
print(f"  MAE: {results_adapts['mae']:.4f}")
print(f"  Coverage: {results_adapts['coverage']*100:.1f}%")

print(f"\nImprovement:")
print(f"  MAE reduction: {((baseline_mae - results_adapts['mae'])/baseline_mae)*100:.1f}%")
print(f"  Coverage difference: {(results_adapts['coverage'] - baseline_coverage)*100:.1f} percentage points")

# Experiment 2: Window Size
print("\n" + "="*70)
print("EXPERIMENT 2: Window Size Sensitivity")
print("="*70)

window_sizes = [5, 10, 15, 20, 30, 50]
window_results = []

for ws in window_sizes:
    print(f"\nTesting window_size={ws}...")
    fm_fresh = BaselineFM(hidden_dim=128, pred_len=24)
    adapts_test = AdapTS(foundation_model=fm_fresh, seq_len=96, pred_len=24, 
                         window_size=ws, alpha_ridge=1.0)
    res = adapts_test.evaluate_online(test_data[:500], verbose=False)
    window_results.append({'window_size': ws, 'mae': res['mae'], 'coverage': res['coverage']})
    print(f"  MAE: {res['mae']:.4f}, Coverage: {res['coverage']*100:.1f}%")

# Experiment 3: Alpha (Ridge regularization)
print("\n" + "="*70)
print("EXPERIMENT 3: Alpha (Ridge Regularization) Sensitivity")
print("="*70)

alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
alpha_results = []

for alpha in alphas:
    print(f"\nTesting alpha={alpha}...")
    fm_fresh = BaselineFM(hidden_dim=128, pred_len=24)
    adapts_test = AdapTS(foundation_model=fm_fresh, seq_len=96, pred_len=24, 
                         alpha_ridge=alpha, window_size=20)
    res = adapts_test.evaluate_online(test_data[:500], verbose=False)
    alpha_results.append({'alpha': alpha, 'mae': res['mae'], 'coverage': res['coverage']})
    print(f"  MAE: {res['mae']:.4f}, Coverage: {res['coverage']*100:.1f}%")

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Comparison bar chart
ax = axes[0, 0]
methods = ['Baseline\n(FM+Conformal)', 'AdapTS']
maes = [baseline_mae, results_adapts['mae']]
coverages = [baseline_coverage*100, results_adapts['coverage']*100]

x = np.arange(len(methods))
width = 0.35
ax.bar(x - width/2, maes, width, label='MAE', color='steelblue')
ax2 = ax.twinx()
ax2.bar(x + width/2, coverages, width, label='Coverage (%)', color='coral')

ax.set_ylabel('MAE', fontsize=11)
ax2.set_ylabel('Coverage (%)', fontsize=11)
ax.set_title('AdapTS vs Baseline Conformal Prediction', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Plot 2: Window size effect
ax = axes[0, 1]
ws_list = [r['window_size'] for r in window_results]
mae_list = [r['mae'] for r in window_results]
cov_list = [r['coverage']*100 for r in window_results]

ax.plot(ws_list, mae_list, 'o-', linewidth=2, markersize=8, label='MAE', color='steelblue')
ax2 = ax.twinx()
ax2.plot(ws_list, cov_list, 's-', linewidth=2, markersize=8, label='Coverage (%)', color='coral')

ax.set_xlabel('Window Size', fontsize=11)
ax.set_ylabel('MAE', fontsize=11)
ax2.set_ylabel('Coverage (%)', fontsize=11)
ax.set_title('Effect of Window Size', fontsize=12, fontweight='bold')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Plot 3: Alpha effect
ax = axes[1, 0]
alpha_list = [r['alpha'] for r in alpha_results]
mae_list = [r['mae'] for r in alpha_results]
cov_list = [r['coverage']*100 for r in alpha_results]

ax.semilogx(alpha_list, mae_list, 'o-', linewidth=2, markersize=8, label='MAE', color='steelblue')
ax2 = ax.twinx()
ax2.semilogx(alpha_list, cov_list, 's-', linewidth=2, markersize=8, label='Coverage (%)', color='coral')

ax.set_xlabel('Alpha (Ridge Regularization)', fontsize=11)
ax.set_ylabel('MAE', fontsize=11)
ax2.set_ylabel('Coverage (%)', fontsize=11)
ax.set_title('Effect of Alpha (Regularization)', fontsize=12, fontweight='bold')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Plot 4: Summary table
ax = axes[1, 1]
ax.axis('off')

summary_text = f"""
SUMMARY OF EXPERIMENTS

Baseline (FM + Conformal only):
  MAE: {baseline_mae:.4f}
  Coverage: {baseline_coverage*100:.1f}%

AdapTS (Full System):
  MAE: {results_adapts['mae']:.4f}
  Coverage: {results_adapts['coverage']*100:.1f}%
  
Improvement: {((baseline_mae - results_adapts['mae'])/baseline_mae)*100:.1f}% MAE reduction

Best Window Size: {min(window_results, key=lambda x: x['mae'])['window_size']}
  MAE: {min(window_results, key=lambda x: x['mae'])['mae']:.4f}

Best Alpha: {min(alpha_results, key=lambda x: x['mae'])['alpha']}
  MAE: {min(alpha_results, key=lambda x: x['mae'])['mae']:.4f}
"""

ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('experiments_results.png', dpi=300, bbox_inches='tight')
print("\n" + "="*70)
print("Saved: experiments_results.png")
print("="*70)