import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Specify the directory containing the CSV files
DATA_DIRECTORY = './CICIDS2017'

def load_and_preprocess_data():
    # Initialize an empty list to store individual dataframes
    data_frames = []
    
    # Iterate through all files in the directory
    for file in os.listdir(DATA_DIRECTORY):
        if file.endswith(".csv"):  # Check if the file is a CSV
            file_path = os.path.join(DATA_DIRECTORY, file)
            temp_df = pd.read_csv(file_path)
            data_frames.append(temp_df)
    
    # Concatenate all dataframes into a single dataframe
    full_data = pd.concat(data_frames, ignore_index=True)
    
    full_data.fillna(method='ffill', inplace=True)  # Handling missing values
    
    X = full_data.drop('Label', axis=1)  # Adjust 'Label' to your target column if different
    y = full_data['Label']              # Adjust 'Label' to your target column if different
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test
