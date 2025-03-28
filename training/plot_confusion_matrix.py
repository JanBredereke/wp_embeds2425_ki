import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, \
    ConfusionMatrixDisplay
from torch import nn


def plot_confusion_matrix(model, test_data, categories, device="cpu"):
    """
    @author <Serkay-Günay Celik>

    Plots a confusion matrix for the predictions made on test data.

    Args:
        model (nn.Module): Trained PyTorch model.
        test_data (list[tuple[str, torch.Tensor]]): Test data as (label, tensor).
        categories (list[str]): List of category names.
        device (str): Device to use ("cuda" or "cpu").
    """
    model.eval()  # Set the model to evaluation mode (disables dropout and batch normalization updates)

    # Collect true labels and predicted labels
    true_labels = []
    predicted_labels = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for i, (test_sample, true_label) in enumerate(test_data):
            # Move test sample to the specified device and add a batch dimension
            test_tensor = test_sample.to(device).unsqueeze(0)

            # Convert the true label from text to index
            true_label_index = categories.index(true_label)
            true_labels.append(true_label_index)

            # Get the model prediction
            output = model(test_tensor)  # Forward pass
            predicted_index = torch.argmax(output, dim=1).item()  # Get index of highest probability
            predicted_labels.append(predicted_index)

    # Compute the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=range(len(categories)))

    # Display the confusion matrix as a heatmap
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)  # Use a blue color map and rotate x-axis labels
    plt.title("Confusion Matrix")  # Set the plot title
    plt.xlabel("Predicted Label")  # Label for the x-axis
    plt.ylabel("True Label")  # Label for the y-axis
    plt.show()  # Display the plot
