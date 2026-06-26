import pandas as pd

try:
    df = pd.read_excel('results/experiment_results.xlsx', sheet_name='experiment_summary')
    print("=== Experiment Summary ===")
    print(df.to_string())
except Exception as e:
    print("Could not read summary:", e)
