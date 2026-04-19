import pandas as pd

# Load the dataset
file_path = "Data/v1_1_pilot_benchmark.csv" 
df = pd.read_csv(file_path)

# Drop the 'is_ambiguous' column
if 'is_ambiguous' in df.columns:
    df = df.drop(columns=['is_ambiguous'])
    print("Column 'is_ambiguous' removed successfully.")

# Save the updated dataset back to CSV
df.to_csv(file_path, index=False)
print("Updated CSV saved.")