import pandas as pd
df = pd.read_csv('data/golden/cleaned/t5_base_joint_golden.csv')
sarcasm = df[df['subtype'] == 'sarcasm']
print(f"Sarcasm samples: {len(sarcasm)}")
print(f"Sarcasm flips: {sarcasm['human_flipped_strict'].sum()}")
print(f"Sarcasm flip rate: {sarcasm['human_flipped_strict'].mean()*100:.1f}%")