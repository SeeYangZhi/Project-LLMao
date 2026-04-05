"""
dry_run_eval.py
Creates a small synthetic dataset for quick pipeline sanity-checking.
Usage:
    python scripts/dry_run_eval.py
    python scripts/eval_pipeline.py --input dry_run_input.csv --output dry_run_results.csv --skip_judge
"""
import pandas as pd
def main():
    data = [
        {
            "id": 1,
            "input": "Oh great, the app crashed again on launch day",
            "output": "The app crashed on launch day",
            "subtype": "sarcasm",
        },
        {
            "id": 2,
            "input": "Wow, politicians promise to fix roads, what a surprise",
            "output": "Politicians promised to fix roads",
            "subtype": "irony",
        },
        {
            "id": 3,
            "input": "Yeah because waiting 3 hours is totally fine",
            "output": "Waiting 3 hours is inconvenient",
            "subtype": "overstatement",
        },
        {
            "id": 4,
            "input": "Sure, because that always works out perfectly",
            "output": "That approach does not always work",
            "subtype": "understatement",
        },
        {
            "id": 5,
            "input": "Oh brilliant, another meeting that could be an email",
            "output": "Another meeting was scheduled unnecessarily",
            "subtype": "satire",
        },
        {
            "id": 6,
            "input": "Right, because that worked SO well last time",
            "output": "That did not work well last time",
            "subtype": "rhetorical_question",
        },
        {
            "id": 7,
            "input": "Oh fantastic, the server is down again",
            "output": "Wow the server is incredibly down again as always",
            "subtype": "sarcasm",
        },
        {
            "id": 8,
            "input": "Yeah sure, deadlines are totally just suggestions",
            "output": "Oh yeah deadlines are just suggestions right",
            "subtype": "irony",
        },
    ]
    df = pd.DataFrame(data)
    output_path = "dry_run_input.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path}")
    print(df.to_string(index=False))
    print(f"\nNow run:")
    print(f"  python scripts/eval_pipeline.py --input {output_path} --output dry_run_results.csv --skip_judge")
if __name__ == "__main__":
    main()