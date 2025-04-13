import os
import pandas as pd
import numpy as np
folder_path = 'src\latest_ai_development\Merging'
output_file = 'merged_bank_statements.csv'
dataframes = []

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        try:
            df = pd.read_csv(file_path)
            # df['Source_File'] = filename 
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Combine all DataFrames into one
merged_df = pd.concat(dataframes, ignore_index=True)
merged_df.to_csv(os.path.join(folder_path, output_file), index=False)
print(f"✅ Merged {len(dataframes)} CSV files into '{output_file}'")
