
from torch.utils.data import Subset, random_split, DataLoader
from src.dataset import MyDataset

def create_dataloaders_random(path, transformations, batch_size, split: list = [0.9, 0.05, 0.05], num_workers: int = 4):
    """
    Organize and create the dataloaders for train, valid and test using our datset objet. The split is RANDOM! 
    
    Args:
        path (str): Path to the base folder that contains images.
        transformations (callable): Transformations to apply to each image.
        batch_size (int): Batch size.
        split (list): List of ratios to split the dataset into train, validation, and test.
        num_workers (int): Number of workers to use in the DataLoader.
    
    Returns:
        tr_dl (DataLoader): Training dataloader.
        val_dl (DataLoader): Validation dataloader.
        test_dl (DataLoader): Test dataloader.
        class_names (dict): Mapping of class names to their numeric labels.
    """
    dataset = MyDataset(path = path, transformations=transformations)
    #Calculate the len for each split (train, valid, test)
    dataset_len = len(dataset)
    train_len = int(dataset_len * split[0])
    val_len = int(dataset_len * split[1])
    test_len = int(dataset_len * split[2])

    #Create the splits of dataset
    tr_ds, val_ds, test_ds = random_split(dataset, lengths=[train_len, val_len, test_len])

    #Create the dataloaders
    tr_dl = DataLoader(dataset=tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers) 
    #Avoid shuffle for replicability in both val and test 
    val_dl = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    #test images are evaluated 1 by 1  
    test_dl = DataLoader(dataset=test_ds, batch_size=1, shuffle=False, num_workers=num_workers)  

    #Return dataloaders for each split andthe class_names with its IDs (keys are the name, value the ID)
    return tr_dl, val_dl, test_dl, dataset.class_names


def stratified_dataset_split(dataset: MyDataset, split: list = [0.9, 0.05, 0.05]):
    """
    Creates stratified subsets of the dataset by maintaining class proportions.
    
    Args:
        dataset (Dataset): The original dataset.
        split (list): List of split ratios for train, validation, and test.
    
    Returns:
        tr_ds (Subset): Subset for the training set.
        val_ds (Subset): Subset for the validation set.
        test_ds (Subset): Subset for the test set."
    """
    #We want to create a dictionary of image indexes for each class
    class_idxs = {}
    for idx, (_, label) in enumerate(dataset):
        if label not in class_idxs:
            class_idxs[label] = []
        class_idxs[label].append(idx)

    #Create the splits for each clas
    train_idxs, val_idxs, test_idxs = [], [], []
    for label, idxs in class_idxs.items():
        n_total = len(idxs)
        n_train = int(n_total * split[0])
        n_val = int(n_total * split[1])
        n_test = int(n_total * split[2])

        #Divide the indexes (idxs is a list of indexes) of each class in our list of indexes per split
        train_idxs.extend(idxs[:n_train])
        val_idxs.extend(idxs[n_train:n_train + n_val])
        test_idxs.extend(idxs[n_train + n_val:n_train + n_val + n_test])

    #Create the subsets using torch utils Subset
    tr_ds = Subset(dataset, train_idxs)
    val_ds = Subset(dataset, val_idxs)
    test_ds = Subset(dataset, test_idxs)

    return tr_ds, val_ds, test_ds


def create_stratified_dataloaders(path, transformations, batch_size: int, split: list = [0.9, 0.05, 0.05], num_workers: int = 4):
    """
    Creates stratified dataloaders for the training, validation, and test sets while maintaining the class distribution.
    
    Args:
        path (str): Path to the dataset.
        transformations (callable): Transformations to apply to the images.
        batch_size (int): Batch size.
        split (list): Ratios for train, validation, and test splits.
        num_workers (int): Number of workers to use in the DataLoader.
    
    Returns:
        tr_dl (DataLoader): Training dataloader.
        val_dl (DataLoader): Validation dataloader.
        test_dl (DataLoader): Test dataloader.
        class_names (dict): Mapping of class names to their numeric labels.
        batch_size (int): Variable batch size to use in the training.
    """
    dataset = MyDataset(path=path, transformations=transformations)
    #Stratified_dataset_split will return subsets of datasets acording our factors
    tr_ds, val_ds, test_ds = stratified_dataset_split(dataset, split)

    #Create the stratified dataloaders
    tr_dl = DataLoader(dataset=tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #Typically test_dl shuffle is false, but in this case we will keep it true for the randomnes on inference visualization
    test_dl = DataLoader(dataset=test_ds, batch_size=1, shuffle=True, num_workers=num_workers)

    #Return dataloaders and classes
    return tr_dl, val_dl, test_dl, dataset.class_names