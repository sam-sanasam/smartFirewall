import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
from utils_tools.data_preprocessing import data_preprocess
# import openai
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


def generate_firewall_rules(json_file_path):
    # Load JSON data
    with open(json_file_path, 'r') as json_file:
        anomaly_data = json.load(json_file)

    # Format data for GenAI model
    prompt = f"Generate firewall rules to mitigate the following anomalies:\n{json.dumps(anomaly_data, indent=4)}"

    # Interact with GenAI
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )

    # Extract and print generated rules
    rules = response.choices[0].text.strip()
    print("Generated Firewall Rules:")
    print(rules)

    return rules


def main(file_path):
    # Load and preprocess dataset (implementation details will depend on your specific functions)
    X_train, X_test, scaler, column_names, original_indices = data_preprocess(file_path)

    # Define model dimensions
    input_dim = X_train.shape[1]
    latent_dim = 32

    # Build and compile models
    generator = build_generator(latent_dim, input_dim)
    discriminator = build_discriminator(input_dim)
    discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    gan = build_gan(generator, discriminator)
    gan.compile(loss='binary_crossentropy', optimizer='adam')

    # Training Parameters
    epochs = 500
    batch_size = 64
    sample_interval = 50

    # Labels for real and fake data
    real_label = np.ones((batch_size, 1))
    fake_label = np.zeros((batch_size, 1))

    # Training Loop
    for epoch in range(epochs):
        # Select a random batch of real network traffic
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_data = X_train[idx]

        # Generate fake network traffic
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_data = generator.predict(noise)

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(real_data, real_label)
        d_loss_fake = discriminator.train_on_batch(fake_data, fake_label)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the generator
        g_loss = gan.train_on_batch(noise, real_label)

        # Output progress
        if epoch % sample_interval == 0:
            print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

    # Save models for later use
    generator.save('generator_model.keras')
    discriminator.save('discriminator_model.keras')

    return generator, discriminator, X_test


if __name__ == '__main__':
    main(file_path="/Users/I748497/Downloads/unsw-data/UNSW_NB15_training-set.csv")
