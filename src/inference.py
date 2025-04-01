import torch
import numpy as np
import cv2
from captum.attr import LayerGradCam, IntegratedGradients
from src.training import move2device


def run_inference(model, device, test_dl):
    """
    Runs inference on a test DataLoader and collects predictions, labels, and images.

    Args:
        model: Trained PyTorch model.
        device (str): Device to run the inference (e.g., "cuda").
        test_dl: DataLoader for the test dataset.

    Returns:
        tuple: (images, labels, predictions, accuracy)
    """
    model.eval()
    model.to(device)

    all_images, all_labels, all_preds = [], [], []
    correct = 0

    for batch in test_dl:
        #Get img and lbl and move to device
        img, lbl = move2device(batch, device)
        # Forward pass
        result = model(img)
        pred = torch.argmax(result, dim=1)
        #Check if pred is correct and transform into a number (not a bool tensor!)
        correct += (pred == lbl).sum().item()

        # Collect data
        all_images.extend(img.cpu())  # Convert to CPU for visualization
        all_labels.extend(lbl.cpu().numpy())
        all_preds.extend(pred.cpu().numpy())

    accuracy = (correct / len(test_dl.dataset)) * 100
    return all_images, all_labels, all_preds, accuracy

def calculate_gradcam(model, layer, input_tensor, target_class):
    """
    Computes the GradCAM heatmap input image and label.

    Args:
        model: PyTorch model.
        layer: Target layer for GradCAM.
        input_tensor (torch.Tensor): Single image tensor with shape [1, C, H, W].
        target_class (int): Target class index.
        device: device where to perform the calculation ("cuda", "cpu")

    Returns:
        np.array: Resized GradCAM heatmap.
    """
    
    #Create GradCAM object extractor with LayeredGradCam, model and target layer
    gradcam = LayerGradCam(model, layer)
    #Get the attributes of the image according to the target class
    attr = gradcam.attribute(input_tensor, target=target_class)
    #Detach, move to cpu and transform to numpy
    heatmap = attr.squeeze().cpu().detach().numpy()
    #In the ConvNext, the target layer produces a 7x7 map. We need to make it bigger to place over the org iamge
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    #Normalize the heatmap values. +1e-8 is to avoid div by 0 in case all pixels are equals.
    heatmap_resized = (heatmap_resized - np.min(heatmap_resized)) / (np.max(heatmap_resized) - np.min(heatmap_resized) + 1e-8)
    
    return heatmap_resized

def calculate_integrated_gradients(model, input_tensor, target_class, n_steps=100):
    """
    Computes Integrated Gradients for a given input.

    Args:
        model: PyTorch model.
        input_tensor (torch.Tensor): Single image tensor with shape [1, C, H, W].
        target_class (int): Target class index.
        n_steps (int): Number of steps for IG computation.

    Returns:
        torch.Tensor: Attributions for the input image.
    """
    ig = IntegratedGradients(model)
    attributions = ig.attribute(input_tensor, target=target_class, n_steps=n_steps)
    return attributions

