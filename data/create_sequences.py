import numpy as np

def create_lookback_sequences(scaled_data, lookback=10):
    """
    This function takes scaled time series data and creates sequences of observations
    using a specified lookback period for reinforcement learning. Each observation
    contains 'lookback' number of historical data points, maintaining the temporal order.

    Args:
        scaled_data (numpy array): The scaled time series data as a numpy array.
        lookback (int): The number of historical datapoints to include in each observation.
                        Defaults to 10.

    Returns:
        numpy array: A 3D numpy array of shape (num_sequences, lookback, num_features),
                     where 'num_sequences' is the number of observation sequences
                     created, and 'num_features' is the number of features in each
                     datapoint.
    """
    # Ensure the data has enough rows to create sequences
    if len(scaled_data) < lookback:
        raise ValueError("The dataset is too small to create sequences with the given lookback period.")

    # Initialize an empty list to hold the observation sequences
    sequences = []

    # Loop through the data and create sequences with the specified lookback
    for i in range(len(scaled_data) - lookback + 1):
        # Select the lookback period of historical data points
        sequence = scaled_data[i:i + lookback]
        sequences.append(sequence)

    # Convert the list of sequences into a numpy array
    sequences_array = np.array(sequences)

    return sequences_array

# # Example usage (assuming scaled_data is obtained from the preprocess_data function):
# scaled_data = preprocess_data('financial_data_with_indicators.csv')
# observation_sequences = create_lookback_sequences(scaled_data, lookback=10)
# print(observation_sequences.shape)  # (num_sequences, 10, num_features)
