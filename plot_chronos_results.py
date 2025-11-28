import matplotlib.pyplot as plt
import numpy as np

# Manual data from your terminal output
results = {
    'GOOGL': {'chronos': 0.0786, 'adapts': 0.0686, 'improvement': 12.7},
    'MSFT': {'chronos': 0.0637, 'adapts': 0.0564, 'improvement': 11.5},
    'WMT': {'chronos': 0.0512, 'adapts': 0.0447, 'improvement': 12.7}
}

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

tickers = list(results.keys())
chronos_mae = [results[t]['chronos'] for t in tickers]
adapts_mae = [results[t]['adapts'] for t in tickers]

x = np.arange(len(tickers))
width = 0.35

bars1 = ax.bar(x - width/2, chronos_mae, width, label='Chronos+Conf', alpha=0.8, color='skyblue')
bars2 = ax.bar(x + width/2, adapts_mae, width, label='Chronos+AdapTS+Conf', alpha=0.8, color='lightgreen')

ax.set_xlabel('Dataset', fontsize=12)
ax.set_ylabel('MAE', fontsize=12)
ax.set_title('Chronos (Pre-trained FM) vs Chronos + AdapTS', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(tickers)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

for i, (t, bar1, bar2) in enumerate(zip(tickers, bars1, bars2)):
    imp = results[t]['improvement']
    height = max(bar1.get_height(), bar2.get_height())
    ax.text(i, height * 1.05, f'+{imp:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', color='green')

plt.tight_layout()
plt.savefig('chronos_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved chronos_comparison.png")