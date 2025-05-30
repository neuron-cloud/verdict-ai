import pandas as pd
from features import extract_features
import csv

# Load your sentence dataset
df = pd.read_csv("app/sentence_dataset_full.csv", encoding="latin1")

# Store results here
rows = []

for _, row in df.iterrows():
    label = row['label']
    text = row['sentence']
    features = extract_features(text)
    features['label'] = label
    rows.append(features)

# Create output DataFrame
output_df = pd.DataFrame(rows)

# Reorder columns to match expected format
cols = ['label'] + [col for col in output_df.columns if col != 'label']
output_df = output_df[cols]

# Save as feature_dataset_v2.csv
output_df.to_csv("app/feature_dataset_v2.csv", index=False)
print("✅ Features generated and saved to feature_dataset_v2.csv")
