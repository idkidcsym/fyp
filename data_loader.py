import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import numpy as np

# Specify the directory containing the CSV files
DATA_DIRECTORY = './CICIDS2017'

def load_and_preprocess_data():
    data_frames = []
    for file in os.listdir(DATA_DIRECTORY):
        if file.endswith(".csv"): 
            file_path = os.path.join(DATA_DIRECTORY, file)
            temp_df = pd.read_csv(file_path)
            temp_df.columns = temp_df.columns.str.strip()
            temp_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            temp_df.ffill(inplace=True)  # Forward fill to handle remaining missing values
            data_frames.append(temp_df)
    
    full_data = pd.concat(data_frames, ignore_index=True)
    X = full_data.drop('Label', axis=1)
    y = full_data['Label']

    # Handle non-numeric data if necessary
    X = X.select_dtypes(include=[np.number])
    X.fillna(X.mean(), inplace=True)  # Calculate mean only for numeric columns

    scaler = StandardScaler()
    try:
        X = scaler.fit_transform(X)
    except Exception as e:
        print(f"Error during scaling: {e}")
        return None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test

try:
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print("Data processing complete.")
except Exception as e:
    print(f"An error occurred: {e}")
