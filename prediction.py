from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

from app_model.main import prepare_anomaly_df, reverse_normalize
from utils_tools.data_preprocessing import data_preprocess  # Ensure this function handles new datasets properly


def load_models(
        generator_path='/Users/I748497/IdeaProjects/smart-firewall-rule/model/generator_model.h5',
        discriminator_path='/Users/I748497/IdeaProjects/smart-firewall-rule/model/discriminator_model.h5'):
    # Load the pre-trained models
    generator = load_model(generator_path)
    discriminator = load_model(discriminator_path)
    return generator, discriminator


def predict_anomalies(generator, discriminator, X_test, latent_dim, threshold=0.5):
    # Generate synthetic data using the Generator
    noise = np.random.normal(0, 1, (X_test.shape[0], latent_dim))
    fake_data = generator.predict(noise)

    # Predict using the Discriminator
    scores = discriminator.predict(X_test)
    scores = scores.flatten()  # Make sure it's 1D

    # Detect anomalies based on the threshold
    anomalies = X_test[scores < threshold]

    return scores, anomalies


def main(new_file_path):
    features = ['sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'swin', 'dwin', 'smean', 'dmean', 'trans_depth',
                'response_body_len']

    # Check if column_names is a 1D list
    if not isinstance(features, list):
        column_names = features.flatten().tolist()  # Flattening in case it's mistakenly structured as a matrix

    print(f"Column Names: {features}")
    print(f"Length of Column Names: {len(features)}")


    threshold = 0.5
    # Load the saved models
    generator, discriminator = load_models()

    # Load and preprocess the new dataset
    X_train, X_test, scaler, column_names, original_indices, features_col = data_preprocess(new_file_path)
    print( f"The column name is {column_names}" )

    # Set latent dimension (should be the same as used during training)
    latent_dim = 32

    # Predict anomalies on the new dataset
    scores, anomalies = predict_anomalies(generator, discriminator, X_test, latent_dim)

    print(f"Detected anomalies in new dataset: {anomalies.shape[0]}")

    anomaly_indices = np.where(scores < threshold)[0]
    print(f" The anomaly_indices is : {len(anomaly_indices)}")


    # checking the total length
    print(f"thy is  {X_test} and \n {features}")
    original_test_data = np.array(X_test)

    # Convert the numpy ndarray tp Panda DF
    original_test_df = pd.DataFrame(original_test_data)
    print(f"The original_test_df is {original_test_df}")

    valid_anomaly_indices = [i for i in anomaly_indices if i < original_test_data.shape[0]]
    anomalies_detected_df = original_test_df.iloc[valid_anomaly_indices]
    print(f"valid_anomaly_indices {anomalies_detected_df}")

    # Create DataFrame
    # original_test_df = pd.DataFrame(anomalies_detected_df, columns=features_col)
    # print(f"valid_anomaly_indices {original_test_df}")

    # # Extract and de-normalize anomalies
    raw_anomalies = reverse_normalize(anomalies_detected_df, scaler)
    print(f" The raw_anomalies is : {raw_anomalies}")

    # Prepare DataFrame for anomalies
    # anomaly_df = prepare_anomaly_df(raw_anomalies, original_indices[anomaly_indices], column_names)



    return anomalies


if __name__ == '__main__':
    new_file_path = "path to the input data/UNSW_NB15_testing-set.csv"
    anomalies = main(new_file_path)
