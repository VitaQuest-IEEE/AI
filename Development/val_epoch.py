import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision
from PIL import Image


def get_val_loss(model, test_dataloader, device, criterion):
    epoch_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    model.eval()  # Set the model to evaluation mode (disables dropout, etc.)
    
    with torch.no_grad():  # Disable gradient calculations
        for i, (images, labels) in enumerate(test_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            out = model(images)
            loss = criterion(out, labels)
            epoch_loss += loss.item()
            
            # Calculate accuracy
            _, y_true = torch.max(labels, 1)  #from one-hot encode to ordinal
            _, preds = torch.max(out, 1)  # Get the predicted class
            correct_predictions += (preds == y_true).sum().item()
            total_predictions += labels.size(0)
    
    # Compute average loss and accuracy
    epoch_loss /= len(test_dataloader)
    accuracy = correct_predictions / total_predictions * 100  # Convert to percentage
        
    return epoch_loss, accuracy
