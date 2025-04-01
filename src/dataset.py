import os 
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

class MyDataset(Dataset):
    """
    Custom dataset to load images from a structured folders organized by class
    
    Args:
    path (str): Base path where the images are organized in subfolders by class.
    transformations: Transformations to apply to each image (for example, from torchvision.transforms).
    """
    def __init__(self, path, transformations=None):
        #Apply img transformations
        self.transformations = transformations
        #Load all images in a path
        self.img_paths = [im_path for im_path in sorted(glob(f"{path}/*/*"))]
        #Dictionaries and counters for classes and count per class
        self.class_names = {}
        self.class_counts = {}
        count = 0

        for idx, img_path in enumerate(self.img_paths):
            #Get the class name
            class_name = self.get_class(img_path)
            #Check if the class exist already and if not appends it to the class name dict
            if class_name not in self.class_names:
                self.class_names[class_name] = count
                self.class_counts[class_name] = 1
                count += 1
            #If it exist, increase the counter for that class
            else:
                self.class_counts[class_name] += 1

    def get_class(self, path) -> str:
        """
        Return the name of the class based on its path
        """
        return os.path.dirname(path).split("/")[-1]
    
    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset
        """
        return len(self.img_paths)
    
    def __getitem__(self, idx: int):
        """
                Gets the image and its corresponding tag from the index.
        Args:

            idx (int): index of the element in the dataset.
        Returns:

            tuple: (image, label) where image is a PIL object possibly transformed. 
                   and the label is the numeric value corresponding to the class.
        """
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.class_names[self.get_class(img_path)]

        #If there are tarnsformation, apply it to return the image with them.
        if self.transformations is not None:
            image = self.transformations(image)
        return image, label