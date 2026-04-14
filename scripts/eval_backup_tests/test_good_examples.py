import pandas as pd

df = pd.read_csv("results/golden/t5_base_joint_merged.csv")

print("=== SUCCESS CASES ===\n")
success = df[(df['human_flipped_strict'] == 1) & (df['similarity'] > 0.85)].head(5)
for _, row in success.iterrows():
    print(f"INPUT:  {row['input']}")
    print(f"OUTPUT: {row['output']}")
    print(f"Subtype: {row['subtype']}, Similarity: {row['similarity']:.3f}\n")

print("=== FAILURE CASES ===\n")
failure = df[df['human_flipped_strict'] == 0].head(5)
for _, row in failure.iterrows():
    print(f"INPUT:  {row['input']}")
    print(f"OUTPUT: {row['output']}")
    print(f"Subtype: {row['subtype']}, Similarity: {row['similarity']:.3f}\n")

print("=== SATIRE FAILURES ===\n")
satire_fail = df[(df['subtype'] == 'satire') & (df['human_flipped_strict'] == 0)].head(5)
for _, row in satire_fail.iterrows():
    print(f"INPUT:  {row['input']}")
    print(f"OUTPUT: {row['output']}")
    print(f"Similarity: {row['similarity']:.3f}\n")