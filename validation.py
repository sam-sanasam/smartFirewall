from model.gan_model import gan_model_pretrain, build_discriminator
from utils_tools.data_preprocessing import data_preprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main(file_path):

    X_train, X_test, mean, std, index = data_preprocess(file_path)
    test_input = X_test.shape[0]
    print(test_input)
    input_dim = X_train.shape[1]
    discriminator = build_discriminator(input_dim)

    # Evaluate anomalies in test set
    scores = discriminator.predict(X_test)
    print(scores)
    # Make sure scores is a 1-dimensional array
    scores = scores.flatten()

    # Define a threshold to classify anomalies
    threshold = 0.5
    anomalies = X_test[scores < threshold]

    # Plotting the results
    # plt.figure(figsize=(10, 6))
    # sns.histplot(scores, bins=50, kde=True)
    # plt.title("Anomaly Scores")
    # plt.xlabel("Discriminator Score")
    # plt.ylabel("Frequency")
    # plt.show()

    print(f"Number of anomalies detected: {len(anomalies)}")
    print(f"Number of anomalies detected: {anomalies}")
    # normal_data = X_test[~anomalies.index]  # Assuming anomalies have a boolean index
    #
    # mean_anomalies = np.mean(anomalies)
    # mean_normal = np.mean(normal_data)
    #
    # print(f"Mean of Anomalies: {mean_anomalies}")
    # print(f"Mean of Normal Data: {mean_normal}")


if __name__ == '__main__':
    main(file_path="/Users/I748497/Downloads/unsw-data/UNSW_NB15_training-set.csv")
