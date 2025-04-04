from utils_tools.data_preprocessing import data_preprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization


# Generator Model
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


# Discriminator Model
def build_discriminator(input_dim):
    model = Sequential([
        Dense(128, input_dim=input_dim),
        LeakyReLU(alpha=0.2),
        Dense(64),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid')
    ])
    return model


# GAN Model (Generator + Discriminator)
def build_gan(generator, discriminator):
    model = Sequential([generator, discriminator])
    return model


def generator_model(latent_dim, input_dim):
    generator_func = build_generator(latent_dim, input_dim)
    return generator_func


def discriminator_models(input_dim):
    discriminator_func = build_discriminator(input_dim)
    return discriminator_func


def gan_model_pretrain(file_path):
    #  Getting the X train and test value
    X_train, X_test, mean, std, original_indices = data_preprocess(file_path)

    # Set random seed for reproducibility
    tf.random.set_seed(42)

    # Set dimensions
    input_dim = X_train.shape[1]
    print(input_dim)

    latent_dim = 32

    # Instantiate models
    generator = generator_model(latent_dim, input_dim)
    discriminator = discriminator_models(input_dim)

    print(f"the generator is {generator} and discriminator is {discriminator}")
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


def predict_anomalies(generator, discriminator, X_test, threshold=0.5):
    latent_dim = 32
    # Generate predictions for X_test
    noise = np.random.normal(0, 1, (X_test.shape[0], latent_dim))
    fake_data = generator.predict(noise)

    # Predict using the Discriminator
    scores = discriminator.predict(X_test)
    scores = scores.flatten()  # Make sure it's 1D

    # Detect anomalies based on the threshold
    anomalies = X_test[scores < threshold]

    return anomalies


if __name__ == '__main__':
    generator, discriminator, X_test = gan_model_pretrain(file_path="/Users/I748497/Downloads/unsw-data/UNSW_NB15_training-set.csv")
    anomalies = predict_anomalies(generator, discriminator, X_test)

    print(f"Detected anomalies: {anomalies.shape[0]}")

# if __name__ == '__main__':
#     gan_model_pretrain(file_path="/Users/I748497/Downloads/unsw-data/UNSW_NB15_training-set.csv")
