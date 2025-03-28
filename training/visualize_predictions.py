import matplotlib.pyplot as plt
import torch
from torch import nn


def visualize_predictions(model, test_data, categories, num_samples=5, device="cpu"):
    """
    @author <Serkay-Günay Celik>

    Visualizes predictions for a few test samples, displaying both numerical outputs and probability distributions.

    Args:
        model (nn.Module): Trained PyTorch model.
        test_data (list[tuple[str, torch.Tensor]]): List of test samples as (label, tensor).
        categories (list[str]): List of category names.
        num_samples (int): Number of test samples to visualize.
        device (str): Device to use ("cuda" or "cpu").
    """
    model.eval()  # Set the model to evaluation mode (disables dropout and batch norm updates)

    # Ensure we don't visualize more samples than available
    samples_to_visualize = min(len(test_data), num_samples)

    # Disable gradient calculations for inference
    with torch.no_grad():
        for i in range(samples_to_visualize):
            # Retrieve the test sample and true label
            test_sample, true_label = test_data[i]

            # Move the test sample to the specified device and add a batch dimension
            test_tensor = test_sample.to(device).unsqueeze(0)

            # Convert true label from text to an index
            true_label_index = categories.index(true_label)

            # Model prediction
            output = model(test_tensor)  # Forward pass through the model

            # Apply softmax to convert logits into probabilities
            probabilities = torch.softmax(output, dim=1).squeeze()

            # Get the index of the class with the highest probability
            predicted_index = torch.argmax(probabilities).item()
            predicted_label = categories[predicted_index]

            # Print the results
            print(f"Sample {i + 1}:")
            print(f"  True Label: {true_label} (Index {true_label_index})")
            print(f"  Predicted Label: {predicted_label} (Index {predicted_index})")
            print(f"  Probabilities: {probabilities}")

            # Visualization: Plot probability distribution
            plt.figure(figsize=(8, 4))
            plt.bar(categories, probabilities.cpu().numpy(), color='blue', alpha=0.7)
            plt.title(f"Sample {i + 1}: True Label = {true_label}, Predicted = {predicted_label}")
            plt.xlabel("Categories")
            plt.ylabel("Probability")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
