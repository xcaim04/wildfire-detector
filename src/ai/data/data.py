import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def load_and_process_data():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "forestfires.csv")

    # Load dataset
    df = pd.read_csv(csv_path)

    # Map categorical text features to numerical values
    months_map = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    days_map = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6, "sun": 7}

    df["month"] = df["month"].map(months_map)
    df["day"] = df["day"].map(days_map)

    # Create binary target classification label (1 for fire, 0 for no fire)
    df["fire"] = (df["area"] > 0).astype(np.float32)

    # Separate features and target
    X = df.drop(columns=["area", "fire"]).values.astype(np.float32)
    y = df["fire"].values.reshape(-1, 1).astype(np.float32)

    # Split into train (80%) and test (20%) sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Normalize features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler next to the data script for future API use
    scaler_path = os.path.join(current_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    # Convert numpy arrays to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_process_data()
    print("Data loading test successful.")
    print(f"Train features shape: {X_train.shape}")
    print(f"Train labels shape: {y_train.shape}")
