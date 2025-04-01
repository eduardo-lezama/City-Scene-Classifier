import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import confusion_matrix
from torchvision import transforms as T
from src.dataset import MyDataset

def tensor2image(tensor, img_type = "rgb"):
    """
    Converts a PyTorch tensor back into a visualizable image (denormalized).

    Args:
        tensor (torch.Tensor): Tensor image with shape [C, H, W].
        img_type (str): Type of the image ("rgb" or "gray").

    Returns:
        np.array: Image ready for matplotlib visualization with shape [H, W, C].
    """
    #Tensor images in pytorch are expressed in [C x H x W]
    #We need to transform the tensors back to images
    gray_transforms = T.Compose([
        #First we invert the std
        T.Normalize(mean = [0.], std = [1/0.5]),
        #Then we invert the mean
        T.Normalize(mean = [-0.5], std = [1])
    ])

    rgb_transforms = T.Compose([
        #First we invert the std
        T.Normalize(mean = [0.,0.,0.], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
        #Then we invert the mean (sum the mean, because is the opposite in the normalization, substract)
        T.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])
    ])

    invert_transforms = rgb_transforms if img_type.lower() == "rgb" else gray_transforms
    #invert_transform --> applies inverse transformations to the tensored image and multiply by 255 to get the px value
    #.detach() --> Disconnect the tensor from calculating gradients
    #.squeeze() --> Deletes dimensions with 1, for instance in gray images
    #.cpu() --> move the tensor to the CPU if it was located in the GPU
    #.permut(1,2,0) --> changes axes of the tesnor (C, H, W,) for the ones expected by matplotlib (H, W, C)
    #.numpy() --> converts the array to a numpy array
    #.astype(np.uint8) --> Converts the values to the 8bit format of images
    #The result  is a an image ready to be visualized by matplotlib (denormalized, H,W,C, 224x224)
    return (invert_transforms(tensor) * 255).detach().squeeze().cpu().permute(1,2,0).numpy().astype(np.uint8)
    
    #return (invert_transforms(tensor) * 255).detach().squeeze().cpu().permute(1,2,0).numpy().astype(np.uint8) if img_type == "gray" else (invert_transforms(tensor) * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8)


def plot_images(dataset, n_imgs: int, rows: int, img_type: str = None, class_names = None):
    """
    Plots a grid of random images from the dataset along with their true labels.

    Args:
        dataset (Dataset): Dataset object from which to sample images.
        n_imgs (int): Number of images to display in the grid.
        rows (int): Number of rows in the grid.
        img_type (str): Type of the image ("rgb" or "gray").
        class_names (dict, optional): Mapping of numeric labels to class names.
    """
    #Check if the input has a img_type otherwise rise an assertError
    assert img_type.lower() in ["rgb", "gray"], "Missing image type (rgb or gray)"
    #Select the color map for the type of image
    if img_type == "rgb":
        #If the images is RGB select viridis color map
        cmap = "viridis"
    else:
        cmap = img_type
    #Create the figure where the images will be displayed
    plt.figure(figsize=(20,10))
    #Create a list of random indexes of images from the dataset and input "n_imgs"
    rndm_idx = [random.randint(0, len(dataset) -1) for _ in range(n_imgs)]
    
    for i, idx in enumerate(rndm_idx):
        img_tensor, label = dataset[idx]
        #Create one image inside figure (1 position inside the grid)
        plt.subplot(rows, n_imgs // rows, i + 1)
        if img_type:
            plt.imshow(tensor2image(tensor=img_tensor, img_type=img_type), cmap=cmap)
        #If there is no type specified, use the default value which is "rgb"
        else:
            plt.imshow(tensor2image(tensor=img_tensor))
        plt.axis("off")
        if class_names is not None:
            #If class_name is defined, use it
            plt.title(f"True L --> {class_names[int(label)]}")
        else:
            #Else, use the true label
            plt.title(f"True L --> {label}")

def class_distribution(dataset_path: str, transformations: T.Compose):
    """
    Plots the distribution of classes in a dataset.

    Args:
        dataset_path (str): Path to the dataset.
        transformations (torchvision.transforms.Compose): Transformations applied to the images.

    Returns:
        None: Displays a bar plot showing the class distributions.
    """
    dataset = MyDataset(path=dataset_path, transformations=transformations)
    
    #Graphic parameters for text
    width = 0.8
    text_width = 0.06
    text_height = 2
    class_counts = dataset.class_counts
    class_names = list(dataset.class_names.keys())
    counts = list(class_counts.values())

    fig, ax = plt.subplots(figsize= (20,10))
    #Return evenly spaced values from a range, for the bar graph
    idxs = np.arange(len(counts))
    #Create the bars
    ax.bar(idxs, counts, width, color="blue")

    #Graphic config
    # Set axis X title
    ax.set_xlabel("Class names", color=("black"))
    # Set the ticks in axis X and its labels
    ax.set(xticks=idxs, xticklabels=class_names)
    # Set a parameter rotation for the axis X labels
    ax.tick_params(axis="x", rotation=45)
    #Set axis Y label
    ax.set_ylabel("Frequency", color="black")
    #Set title for the graph
    ax.set_title("Classes distribution")
    
    #Adds a text over each column
    for i, val in enumerate(counts):
        ax.text(i - text_width, val + text_height, str(val), color="blue")

def plot_confusion_matrix(preds, labels, class_names):
    """
    Plots the confusion matrix for the predictions and true labels.

    Args:
        preds (list): Predicted class indices.
        labels (list): True class indices.
        class_names (list): List of class names corresponding to indices.
    """
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(15, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.show()

def plot_image_grid(images, labels, preds, class_names, num_images, rows):
    """
    Plots a grid of images with predicted and true labels.

    Args:
        images (list): List of image tensors.
        labels (list): True labels.
        preds (list): Predicted labels.
        class_names (list): List of class names.
        num_images (int): Number of images to plot.
        rows (int): Number of rows for the grid.
    """
    #Calculate num columns and round up
    cols = (num_images + rows - 1) // rows 

    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = axes.flatten() #Flatten for an easy iteration

    for idx, ax in enumerate(axes):
        if idx >= num_images: #Important, turn extra axes off if fewer images than grid space
            ax.axis("off")
            break
        #Get image, label and pred
        image = tensor2image(images[idx])
        label = labels[idx]
        prediction = preds[idx]
    
        #Set titles
        color = "green" if label == prediction else "red"
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"True L: {class_names[label]}\nPred: {class_names[prediction]}", color=color)
        
    plt.tight_layout()
    plt.show()


def visualize_gradcam(image, heatmap, target_class):
    """
    Overlays the GradCAM heatmap on the input image and visualizes it using Matplotlib.
    
    Args:
        image (torch.Tensor): Input tensor (single image) in [C, H, W] format.
        heatmap (np.array): GradCAM heatmap with dimensions [H, W].
        target_class (int): The target class index for the GradCAM.
    
    Returns:
        None: Displays the heatmap overlay directly.
    """
    # Convert tensor to an image, assuming it has been properly normalized
    overlay_image = tensor2image(image)  # Reuses your helper function to transform the input image
    
    # Normalize the heatmap values (avoiding division by zero)
    heatmap_resized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Visualization using Matplotlib
    fig, ax = plt.subplots(figsize=(6, 6))  # Create figure and axes
    ax.imshow(overlay_image)  # Show the original image
    heatmap_img = ax.imshow(heatmap_resized, cmap="viridis", alpha=0.4)  # Overlay heatmap with transparency
    
    # Add a colorbar for the heatmap
    cbar = fig.colorbar(heatmap_img, ax=ax, fraction=0.03, pad=0.05)
    cbar.set_label("Importance", rotation=270, labelpad=15)  # Add a label to the color bar
    
    # Add title and hide axis
    ax.set_title(f"GradCAM for class {target_class}")
    ax.axis("off")
    plt.show()

# def overlay_gradcam(image, heatmap):
#     """
#     Overlays a GradCAM heatmap on the original image.

#     Args:
#         image (np.array): Original image in [H, W, C] format.
#         heatmap (np.array): GradCAM heatmap in [H, W] format.

#     Returns:
#         np.array: Combined image with overlay.
#     """
#     heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
#     heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
#     overlay = cv2.addWeighted(image.astype(np.float32), 0.6, heatmap_color.astype(np.float32), 0.4, 0)
#     return overlay