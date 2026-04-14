import pandas as pd

df = pd.read_csv("results/golden/t5_base_joint_merged.csv")

# STRICT SUCCESS = sarcasm removed AND meaning preserved
strict = df[df['human_strict_success'] == 1].copy()
strict = strict.sort_values('similarity', ascending=False)

print("=== TRUE STRICT SUCCESS CASES ===\n")
for _, row in strict.head(10).iterrows():
    print(f"INPUT:  {row['input']}")
    print(f"OUTPUT: {row['output']}")
    print(f"Subtype: {row['subtype']}, Sim: {row['similarity']:.3f}")
    print(f"Meaning Change: {row.get('meaning_change', 'N/A')}")
    print()