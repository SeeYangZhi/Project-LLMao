import pandas as pd
import numpy as np

print("=" * 80)
print("PERPLEXITY DATA SOURCE VERIFICATION")
print("=" * 80)

# Check what's stored in the results CSV
models = [
    ('t5_base_joint', 'results/t5_base_joint_results.csv'),
    ('bart_base_rl', 'results/bart_base_rl_results.csv'),
]

for name, path in models:
    df = pd.read_csv(path)
    ppl = df['perplexity'].dropna()
    
    print(f"\n{name}:")
    print(f"  N samples: {len(ppl)}")
    print(f"  Mean (what Dashboard shows): {ppl.mean():.1f}")
    print(f"  Median: {ppl.median():.1f}")
    print(f"  Max: {ppl.max():.1f}")
    print(f"  Outliers (>= 10000): {len(ppl[ppl >= 10000])}")
    
    # Filtered mean
    filtered = ppl[ppl < 10000]
    print(f"  Filtered mean (<10000): {filtered.mean():.1f}")

# Golden data
print("\n" + "=" * 80)
print("GOLDEN DATA (140 samples)")
print("=" * 80)

golden = [
    ('t5_base_joint', 'results/golden/t5_base_joint_results.csv', 740.0),
    ('t5_base_control', 'results/golden/t5_base_control_results.csv', 750.6),
    ('bart_base_rl', 'results/golden/bart_base_rl_results.csv', 1794.4),
]

for name, path, evalmd_val in golden:
    df = pd.read_csv(path)
    ppl = df['perplexity'].dropna()
    
    mean_val = ppl.mean()
    filtered = ppl[ppl < 10000]
    filtered_mean = filtered.mean()
    
    print(f"\n{name}:")
    print(f"  CSV Mean (unfiltered): {mean_val:.1f}")
    print(f"  CSV Mean (filtered):   {filtered_mean:.1f}")
    print(f"  EVALUATION.md shows:   {evalmd_val}")
    
    if abs(mean_val - evalmd_val) < 10:
        print(f"  → EVALUATION.md uses UNFILTERED ✓")
    elif abs(filtered_mean - evalmd_val) < 10:
        print(f"  → EVALUATION.md uses FILTERED ✓")
    else:
        print(f"  → NEITHER matches! Check data source.")