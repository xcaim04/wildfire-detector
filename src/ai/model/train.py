import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.ai.data.data import load_and_process_data
from src.ai.model.model import WildfireNet


def train_model() -> None:

    # 1. Load the processed tensors
    X_train, X_test, y_train, y_test = load_and_process_data()

    # 2. Package tensors into PyTorch DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 3. Instantiate model, loss function, and optimizer
    model = WildfireNet(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

    # 4. Training loop
    epochs = 100
    print("Starting training loop...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # 5. Evaluate on test data every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()

            with torch.no_grad():
                test_preds = model(X_test)
                test_loss = criterion(test_preds, y_test).item()

                # Convert probabilities to binary choices (0 or 1) at 0.5 threshold
                binary_preds = (test_preds >= 0.5).float()
                accuracy = (binary_preds == y_test).float().mean().item()

            print(
                f"Epoch [{epoch + 1}/{epochs}] -> Train Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f} | Test Accuracy: {accuracy * 100:.2f}%"
            )

    # 6. Save the trained model weights directly into the current 'model' directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir, "wildfire_model.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Training complete. Weights saved to: {weights_path}")
