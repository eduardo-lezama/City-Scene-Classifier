import torch
import torch.amp
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score
from torchvision.utils import make_grid
from torch.amp import GradScaler

def init_tensorboard_logs(experiment: int, hyperparams: dict):
    """
    Initializes TensorBoard and logs hyperparameters.

    Args:
        experiment (int): Experiment number.
        hyperparams (dict): Dictionary of hyperparameters(keys) and values to log.

    Returns:
        SummaryWriter: TensorBoard writer object.
    """
    #Save the hyperparameters in a text for the current experiment using a writer
    writer = SummaryWriter(f"logs/experiment{experiment}")
    formatted_text = "\n".join([f"{key}: {value}" for key, value in hyperparams.items()])
    writer.add_text("Hyperparameters", formatted_text)
    return writer

def setup_train(model, hyperparams: dict):
    """
    Configures the model, optimizer, and loss function for training.

    Args:
        model: PyTorch model instance.
        hyperparams (dict): Dictionary with training parameters.

    Returns:
        tuple: Model, number of epochs, device, loss function, optimizer.
    """
    return model.to("cuda").train(), hyperparams["epochs"], "cuda", torch.nn.CrossEntropyLoss(), torch.optim.Adam(params=model.parameters(), lr=hyperparams["lr"])
    #Put the model in train() behaviour
    #7hyperparams["epochs"] --> epochs
    #"cuda" --> str pointing to cuda device
    #CrossEntropy as the loss function
    #Adam as the optimizer with its parameters

def move2device(batch, device):
    """
    Moves batches of data (images and labels) to the specified device.

    Args:
        batch (tuple): Batch of data.
        device (str): Device to move the data (e.g., "cuda").

    Returns:
        tuple: Images and labels moved to the device.
    """
    return batch[0].to(device), batch[1].to(device)

def calculate_losses(model, images, labels, loss_fn):
    """
    Calculates predictions, loss, and accuracy for a batch.

    Args:
        model: PyTorch model instance.
        images (torch.Tensor): Batch of images.
        labels (torch.Tensor): Batch of labels.
        loss_fn: Loss function.

    Returns:
        tuple: Loss value, batch accuracy.
    """
    preds = model(images)
    loss = loss_fn(preds, labels)
    accuracy = (torch.argmax(preds, dim=1) == labels).sum().item()
    return loss, accuracy

# def calculate_losses(model, images, labels, loss_fn, epoch_loss, epoch_acc):
#     """
#     Calculates predictions, loss, and accuracy for a batch.

#     Args:
#         model: PyTorch model instance.
#         images (torch.Tensor): Batch of images.
#         labels (torch.Tensor): Batch of labels.
#         loss_fn: Loss function.

#     Returns:
#         tuple: Loss value, batch accuracy.
#     """
    
#     #Generate predictions
#     preds = model(images)
#     #Calculate loss
#     loss = loss_fn(preds, labels)
#     #Updates the total loss and the accumulated accuracy
#     return loss, epoch_loss + loss.item(), epoch_acc + (torch.argmax(preds, dim=1) == labels).sum().item()

### Training and validation loops

def train_one_epoch(model, dataloader, device, loss_fn, optimizer, scaler: GradScaler, writer: SummaryWriter, epoch):
    """
    Performs training for a single epoch.

    Args:
        model: PyTorch model instance.
        dataloader: DataLoader for training data.
        device (str): Device to use.
        loss_fn: Loss function.
        optimizer: Optimizer instance.
        scaler: GradScaler for FP16 precision.
        writer: TensorBoard writer.
        epoch (int): Current epoch number.

    Returns:
        dict: Training loss and accuracy metrics.
    """
    epoch_loss, epoch_acc = 0, 0
    model.train() #Put the model into training state

    for batch in tqdm(dataloader, total=len(dataloader)):
        imgs, lbls = move2device(batch, device)
        #Activate autocast for mixed precision
        with torch.amp.autocast(device):
            loss, accuracy = calculate_losses(model, imgs, lbls, loss_fn)
            #Backprop and optimizer step
            optimizer.zero_grad()
            #Calculate grads
            scaler.scale(loss).backward()
            #Optimizer do an step using those grads
            scaler.step(optimizer)
            #Update dynamic scaler
            scaler.update()
            #Calculate loss and accuracy per batch and accumulate
            epoch_loss += loss.item()
            epoch_acc += accuracy

    #Register training metrics into TensorBoard
    #Mean loss per training batch
    tr_loss_to_track = epoch_loss / len(dataloader)
    #Mean acc. per epoch
    tr_acc_to_track = epoch_acc / len(dataloader.dataset)
    writer.add_scalar("Loss/Train", tr_loss_to_track, epoch)
    writer.add_scalar("Accuracy/Train", tr_acc_to_track, epoch)

    #print(f"TRAINING: Epoch: {epoch + 1} train loss: {tr_loss_to_track:.3f}, train accuracy: {tr_acc_to_track:.3f}")
    return {"train_loss": tr_loss_to_track, "train_accuracy": tr_acc_to_track}

def validate_one_epoch(model, dataloader, device, loss_fn, writer: SummaryWriter, epoch):
    """
    Performs validation for a single epoch and tracks metrics.

    Args:
        model: PyTorch model instance.
        dataloader: DataLoader for validation data.
        device (str): Device to use.
        loss_fn: Loss function.
        writer: TensorBoard writer.
        epoch (int): Current epoch number.

    Returns:
        dict: Validation metrics (loss, accuracy, precision, recall).
    """
    model.eval()
    val_epoch_loss, val_epoch_acc, all_preds, all_labels = 0, 0, [], []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            imgs, lbls = move2device(batch, device)
            #Activate autocast for validation (FP16)
            with torch.amp.autocast(device):
                preds = model(imgs)
                loss = loss_fn(preds, lbls)

            val_epoch_loss += loss.item()
            #From preds (for each image has a tensor of 10 logits), get the total of corrects predictions
            val_epoch_acc += (torch.argmax(preds, dim=1) == lbls).sum().item()
            #Accumulation of preds and labels, to calculate at the end of the val.
            #Calculating precisio and recall per batch could cause problems as there could be batches without
            #certain type of images.
            all_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())

            #Save images from the first batch only, as samples
            if i == 0:
                grid = make_grid(imgs[0:8].cpu(), nrow=2, normalize=True)
                writer.add_image(f"Validation examples/Epoch {epoch+1}", grid, epoch)
                writer.add_text(f"Predictions/Epoch {epoch+1}",
                                f"Predicted: {torch.argmax(preds[:8], dim=1).cpu().numpy()}, "
                                f"Actual: {lbls[:8].cpu().numpy(), epoch}", 
                                epoch + 1
                                )

    #Register training metrics into TensorBoard
    #Mean loss per batch
    val_loss_to_track = val_epoch_loss / len(dataloader) # len(val_dl) is the total number of batches
    #Mean accuracy of the val dataset
    val_acc_to_track = val_epoch_acc / len(dataloader.dataset) #The reason we use dataset is because it is calculated per image
    #Mean precision and recall using all preds and labels from the validation cycle.
    precision  = precision_score(all_preds, all_labels, average="weighted")
    recall  = recall_score(all_preds, all_labels, average="weighted")
    #Write and register the metrics
    writer.add_scalar("Loss/Validation", val_loss_to_track, epoch)
    writer.add_scalar("Accuracy/Validation", val_acc_to_track, epoch)
    writer.add_scalar("Precision/Validation", precision, epoch)
    writer.add_scalar("Recall/Validation", recall , epoch)
    
    return {"val_loss": val_loss_to_track, 
            "val_accuracy": val_acc_to_track, 
            "precision": precision, 
            "recall": recall}