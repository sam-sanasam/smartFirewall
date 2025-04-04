import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.models import load_model
from utils_tools.data_preprocessing import data_preprocess
import json


# Define model-building functions
def build_generator(input_dim, output_dim):
    model = Sequential([
        Dense(64, input_dim=input_dim),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(128),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(output_dim, activation='tanh')
    ])
    return model


def build_discriminator(input_dim):
    model = Sequential([
        Dense(128, input_dim=input_dim),
        LeakyReLU(alpha=0.2),
        Dense(64),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid')
    ])
    return model


def build_gan(generator, discriminator):
    model = Sequential([generator, discriminator])
    return model


# Define anomaly detection and processing
def prepare_anomaly_df(raw_anomalies, original_indices, column_names):
    # Create a DataFrame with original indices and raw anomalies
    anomaly_df = pd.DataFrame(raw_anomalies, index=original_indices, columns=column_names)
    return anomaly_df


def reverse_normalize(normalized_data, scaler):
    # Inverse transform to get the original values from normalized values
    return scaler.inverse_transform(normalized_data)


def save_anomalies_to_json(anomaly_df, file_path='detected_anomalies.json'):
    # Convert DataFrame to dictionary and then to JSON
    anomaly_dict = anomaly_df.to_dict(orient='index')

    # Write JSON to file
    with open(file_path, 'w') as json_file:
        json.dump(anomaly_dict, json_file, indent=4)

    print(f"Detected anomalies saved to {file_path}")


# def generate_firewall_rules(json_file_path):
#     # Load JSON data
#     with open(json_file_path, 'r') as json_file:
#         anomaly_data = json.load(json_file)
#
#     # Format data for GenAI model
#     prompt = f"Generate firewall rules to mitigate the following anomalies:\n{json.dumps(anomaly_data, indent=4)}"
#
#     # Interact with GenAI
#     response = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=prompt,
#         max_tokens=150,
#         temperature=0.7
#     )
#
#     # Extract and print generated rules
#     rules = response.choices[0].text.strip()
#     print("Generated Firewall Rules:")
#     print(rules)
#
#     return rules


def load_models(
        generator_path='/Users/I748497/IdeaProjects/smart-firewall-rule/app_model/generator_model.keras',
        discriminator_path='/Users/I748497/IdeaProjects/smart-firewall-rule/app_model/discriminator_model.keras'):
    # Load the pre-trained models
    generator = load_model(generator_path)
    discriminator = load_model(discriminator_path)
    return generator, discriminator


def main(file_path):
    # Load and preprocess dataset (implementation details will depend on your specific functions)
    X_train, X_test, scaler, column_names, original_indices = data_preprocess(file_path)

    # Define model dimensions
    input_dim = X_train.shape[1]
    latent_dim = 32

    # Predict anomalies
    latent_dim = 32
    noise = np.random.normal(0, 1, (X_test.shape[0], latent_dim))

    # Loading the  GAN Model
    generator, discriminator = load_models()

    scores = discriminator.predict(X_test).flatten()

    print(f"The predicted score is {scores}")

    # Identify anomalies
    threshold = 0.5
    anomalies = X_test[scores < threshold]

    anomaly_indices = np.where(scores < threshold)[0]
    print(f"The anomaly_indices  is {anomalies.shape[0]}")

    # # Extract and de-normalize anomalies
    # raw_anomalies = reverse_normalize(X_test[anomaly_indices], scaler)
    #
    # # Prepare DataFrame for anomalies
    # anomaly_df = prepare_anomaly_df(raw_anomalies, original_indices[anomaly_indices], column_names)
    #
    # # Save anomalies to JSON
    # save_anomalies_to_json(anomaly_df)

    # Generate firewall rules
    # generate_firewall_rules('detected_anomalies.json')


if __name__ == '__main__':
    main(file_path="/Users/I748497/Downloads/unsw-data/UNSW_NB15_testing-set.csv")
