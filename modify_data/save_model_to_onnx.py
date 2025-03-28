import torch


def save_model_to_onnx(model, model_name, input_size):
    """
    @author <Sebastian Schramm>

    Save a PyTorch model to ONNX format.

    Args:
        model (nn.Module): PyTorch model to save.
        model_name (str): Name of the model.
        input_size (tuple): Input size of the model.

    """
    # Set the model to evaluation mode
    model.eval()

    # Create a dummy input tensor with the same size as the model input
    dummy_input = torch.randn(1, *input_size)

    # Define the path where the ONNX model will be saved
    onnx_path = f"trained_models/{model_name}.onnx"

    # Export the model to ONNX format
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )

    print(f"Model saved to {onnx_path}")
    # https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html
