import logging  # Import logging module for handling errors and debugging

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import enums.colors as colors
from training.visualize_predictions import visualize_predictions


def train_models(
        models_list: list[nn.Module],
        train_data: list[tuple[str, torch.Tensor]],
        test_data: list[tuple[str, torch.Tensor]],
        number_of_categories: int,
        categories: list[str],
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        device: str | None = None
) -> list[nn.Module]:
    """
    @author <Serkay-Günay Celik, Sebastian Schramm>

    Trains multiple models using the provided training and test data and visualizes predictions.

    Args:
        models_list (list[nn.Module]): List of models to train.
        train_data (list[tuple[str, torch.Tensor]]): Training dataset as (label, tensor).
        test_data (list[tuple[str, torch.Tensor]]): Test dataset as (label, tensor).
        number_of_categories (int): Number of output classes.
        categories (list[str]): List of category names.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        device (str): Device for training ("cuda" or "cpu").

    Returns:
        list[nn.Module]: List of trained models.
    """

    # Define a function for batch collation to pad tensors to the same length
    def collate_fn(batch):
        """Collate function to pad tensors so they have the same length in a batch."""
        labels, tensors = zip(*batch)

        # Convert labels from strings to numerical indices
        labels = torch.tensor([categories.index(label) for label in labels], dtype=torch.long)

        # Find the maximum length of all tensors in the batch
        max_length = max(tensor.size(0) for tensor in tensors)

        # Pad tensors with zeros to match the maximum length
        padded_tensors = torch.stack(
            [F.pad(tensor, (0, max_length - tensor.size(0))) for tensor in tensors]
        )

        return padded_tensors, labels

    # Determine the device for training (use GPU if available)
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data loaders for training and testing
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Training loop for each model in the provided list
    for model in models_list:
        print(f'\n{colors.CYAN}Start of training {model.__class__.__name__}{colors.COLOR_OFF}')

        try:
            # Move the model to the selected device (CPU/GPU)
            model = model.to(device)

            # Define the loss function and optimizer
            criterion = nn.CrossEntropyLoss()  # Standard loss function for classification tasks
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer for training

            # Training loop for the specified number of epochs
            for epoch in range(epochs):
                model.train()  # Set model to training mode
                total_loss = 0  # Track total loss for the epoch

                # Training phase
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)  # Move data to the correct device
                    optimizer.zero_grad()  # Reset gradients
                    outputs = model(inputs)  # Forward pass through the model
                    loss = criterion(outputs, labels)  # Compute loss
                    loss.backward()  # Backpropagation
                    optimizer.step()  # Update model parameters
                    total_loss += loss.item()  # Accumulate loss

                # Print loss for the epoch
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

                # Validation phase
                model.eval()  # Set model to evaluation mode
                correct = 0
                total = 0

                with torch.no_grad():  # Disable gradient calculation during validation
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)  # Move data to the correct device
                        outputs = model(inputs)  # Forward pass
                        _, predicted = torch.max(outputs, 1)  # Get the index of the highest probability class
                        correct += (predicted == labels).sum().item()  # Count correct predictions
                        total += labels.size(0)  # Track total samples

                # Compute and print validation accuracy
                accuracy = correct / total * 100
                print(f"Validation Accuracy: {accuracy:.2f}%")

            # Visualize predictions after training
            visualize_predictions(model, test_data, categories, num_samples=5, device=device)

        except Exception as e:
            print(e)  # Print any errors encountered during training

    return models_list  # Return the list of trained models
