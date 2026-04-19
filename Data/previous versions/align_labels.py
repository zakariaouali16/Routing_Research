import pandas as pd

# Standardize labels in CSV to match taxonomy exactly
df = pd.read_csv('v0_pilot_benchmark.csv')
label_map = {
    'Course Logistics & Env': 'Course Logistics & Environment Setup',
    'Needs Clarification': 'Clarification Needed'
}
df['label'] = df['label'].replace(label_map)
df.to_csv('v0_pilot_benchmark_standardized.csv', index=False)
print("Saved standardized benchmark to: v0_pilot_benchmark_standardized.csv")