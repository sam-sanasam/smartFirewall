import pandas as pd
import numpy as np
import json

# Sample original test data as numpy array
original_test_data = np.random.rand(35069, 11)  # Example array with shape (35069, 11)

# Define column names (replace these with the actual column names)
column_names = ['sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'swin', 'dwin', 'smean', 'dmean', 'trans_depth', 'response_body_len']

# Convert the numpy array to a Pandas DataFrame
original_test_df = pd.DataFrame(original_test_data, columns=column_names)

# Assuming these are the anomaly indices found in the original DataFrame
# These indices should be validated to ensure they are within the bounds of original_test_df
anomaly_indices = [50, 453, 500]  # Replace with actual indices detected as anomalies

# Check that indices are within bounds
valid_indices = [i for i in anomaly_indices if i < len(original_test_df)]

# Filter the original test data based on valid anomaly indices
anomalies_detected_df = original_test_df.iloc[valid_indices]

# Save anomalies as JSON file
def save_anomalies_to_json(anomalies_df, file_path='detected_anomalies.json'):
    anomalies_dict = anomalies_df.to_dict(orient='index')
    with open(file_path, 'w') as json_file:
        json.dump(anomalies_dict, json_file, indent=4)
    print(f"Detected anomalies saved to {file_path}")

# Example usage
if __name__ == '__main__':
    save_anomalies_to_json(anomalies_detected_df)
