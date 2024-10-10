import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def preprocess_data(file_path):
    """
    Loads financial data from a CSV file, drops NaN values,
    scales the numerical columns, and returns a numpy array.
    
    Parameters:
        file_path (str): Path to the CSV file.
    
    Returns:
        np.ndarray: Scaled data excluding the timestamp column.
    """
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Drop any rows with NaN values
    df = df.dropna()

    # Identify the numerical columns (excluding 'binary_representation')
    numerical_columns = ['open', 'close', 'volume', 'MACD', 'Signal_Line', 'RSI', 'Upper_Band', 'Lower_Band']
    
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler only on the numerical columns and transform them
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Convert the DataFrame to a numpy array
    state_data = df[['binary_representation'] + numerical_columns].to_numpy()

    # Return the scaled DataFrame
    return state_data

def load_financial_data():
    """
    Loads financial data from the CSV file located in the data directory.
    
    Returns:
        np.ndarray: Scaled data from the CSV file.
    """
    # Define the relative path to the CSV file
    file_path = os.path.join(os.path.dirname(__file__), './financial_data.csv')
    
    # Preprocess the data
    return preprocess_data(file_path)

# # Example usage
# if __name__ == "__main__":
#     financial_data = load_financial_data()
#     print(financial_data[:5])  # Display the first 5 rows of the scaled data
