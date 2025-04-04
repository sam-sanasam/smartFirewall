import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split




def data_preprocess(file_path):
    # loading the data "/Users/I748497/Downloads/unsw-data/UNSW_NB15_training-set.csv"
    data = pd.read_csv(file_path)
    column_names = data.columns
    print(data.dtypes)
    data = data.dropna()

    # Select relevant features for network traffic analysis
    features = ['sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'swin', 'dwin', 'smean', 'dmean', 'trans_depth',
                'response_body_len']
    X = data[features]
    features_col = X.columns
    original_indices = X.index

    # Normalize the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # # Normalize features
    # mean = np.mean(X, axis=0)
    # std = np.std(X, axis=0)
    # X_scaled = (X - mean) / std

    # Split the data into training and testing sets
    X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

    return X_train, X_test, scaler, column_names, original_indices , features_col


if __name__ == '__main__':
    file_path = "/Users/I748497/Downloads/unsw-data/UNSW_NB15_training-set.csv"
    data_preprocess(file_path)