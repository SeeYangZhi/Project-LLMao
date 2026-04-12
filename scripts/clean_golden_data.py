"""
Clean human-annotated golden data for eval pipeline.
Standardizes columns and handles different annotation formats.

Usage:
    python scripts/clean_golden_data.py
"""
import pandas as pd
import re
from pathlib import Path

def parse_joint_output(output):
    """Parse 'strategy: X rewrite: Y' format to extract just the rewrite."""
    if pd.isna(output) or str(output).strip() == '':
        return ''
    match = re.search(r'rewrite:\s*(.+)', str(output), re.IGNORECASE)
    return match.group(1).strip() if match else str(output)

def clean_golden_file(input_path, output_path, model_name):
    """Clean a single golden data file."""
    print(f"\nProcessing: {input_path}")
    
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df)} rows")
    print(f"  Original columns: {list(df.columns)}")
    
    # Lowercase column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Standardize annotator columns based on file format
    rename_map = {}
    
    for col in df.columns:
        col_lower = col.lower()
        # T5 joint format: 'a', 'c'
        if col == 'a' and 'output_is_non_sarcastic' not in col_lower:
            rename_map[col] = 'human_sarcasm_removed_1'
        elif col == 'c':
            rename_map[col] = 'human_sarcasm_removed_2'
        # T5 control format: 'a_output_is_non_sarcastic'
        elif 'a_output_is_non_sarcastic' in col_lower:
            rename_map[col] = 'human_sarcasm_removed_1'
        # BART format: 'nguyen checking...', 'andrew checking...'
        elif 'nguyen' in col_lower:
            rename_map[col] = 'human_sarcasm_removed_1'
        elif 'andrew' in col_lower:
            rename_map[col] = 'human_sarcasm_removed_2'
        # Strategy column
        elif 'strategy' in col_lower and col != 'subtype':
            rename_map[col] = 'subtype'
    
    df = df.rename(columns=rename_map)
    print(f"  Renamed columns: {rename_map}")
    
    # Parse joint output format (remove "strategy: X rewrite: " prefix)
    if 'joint' in model_name.lower():
        print("  Parsing joint output format...")
        df['output'] = df['output'].apply(parse_joint_output)
    
    # Handle empty/NA outputs - fill with input
    df['output'] = df['output'].fillna('')
    empty_mask = df['output'].astype(str).str.strip() == ''
    df.loc[empty_mask, 'output'] = df.loc[empty_mask, 'input']
    print(f"  Filled {empty_mask.sum()} empty outputs with input")
    
    # Convert annotator columns to numeric
    for col in ['human_sarcasm_removed_1', 'human_sarcasm_removed_2']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle meaning_change: NaN means preserved (0), 1 means changed
    if 'meaning_change' in df.columns:
        df['meaning_change'] = pd.to_numeric(df['meaning_change'], errors='coerce').fillna(0).astype(int)
    else:
        df['meaning_change'] = 0
    
    # Compute consensus columns
    df['human_flipped_strict'] = (
        (df['human_sarcasm_removed_1'] == 1) & 
        (df['human_sarcasm_removed_2'] == 1)
    ).astype(int)
    
    df['human_flipped_lenient'] = (
        (df['human_sarcasm_removed_1'] == 1) | 
        (df['human_sarcasm_removed_2'] == 1)
    ).astype(int)
    
    df['human_meaning_preserved'] = (df['meaning_change'] == 0).astype(int)
    
    df['human_strict_success'] = (
        (df['human_flipped_strict'] == 1) & 
        (df['human_meaning_preserved'] == 1)
    ).astype(int)
    
    # Ensure subtype column exists
    if 'subtype' not in df.columns:
        df['subtype'] = 'unknown'
    
    # Select and order columns for output
    output_cols = [
        'id', 'input', 'output', 'subtype',
        'human_sarcasm_removed_1', 'human_sarcasm_removed_2',
        'human_flipped_strict', 'human_flipped_lenient',
        'meaning_change', 'human_meaning_preserved', 'human_strict_success'
    ]
    
    df_out = df[[c for c in output_cols if c in df.columns]].copy()
    
    # Save
    df_out.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    print(f"  Final columns: {list(df_out.columns)}")
    
    # Print summary stats
    print(f"\n  Summary:")
    print(f"    Total samples: {len(df_out)}")
    print(f"    Annotator 1 flipped: {df_out['human_sarcasm_removed_1'].sum()}")
    print(f"    Annotator 2 flipped: {df_out['human_sarcasm_removed_2'].sum()}")
    print(f"    Both agree flipped: {df_out['human_flipped_strict'].sum()}")
    print(f"    Meaning changed: {df_out['meaning_change'].sum()}")
    print(f"    Strict success: {df_out['human_strict_success'].sum()}")
    
    return df_out

def main():
    base = Path("data/golden")
    raw_dir = base / "raw"
    clean_dir = base / "cleaned"
    
    # Create output directory
    clean_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each file
    files = [
        ("t5_base_joint_human.csv", "t5_base_joint_golden.csv", "t5_base_joint"),
        ("t5_base_control_human.csv", "t5_base_control_golden.csv", "t5_base_control"),
        ("bart_base_rl_human.csv", "bart_base_rl_golden.csv", "bart_base_rl"),
    ]
    
    for raw_name, clean_name, model_name in files:
        input_path = raw_dir / raw_name
        output_path = clean_dir / clean_name
        
        if not input_path.exists():
            print(f"\nWARNING: {input_path} not found, skipping...")
            continue
        
        clean_golden_file(input_path, output_path, model_name)
    
    print("\n" + "="*60)
    print("CLEANING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()