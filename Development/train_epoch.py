import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision
from PIL import Image


def train_epoch(model, train_dataloader, device, criterion, optimizer):
    epoch_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    model.train()  # Set the model to training mode
    
    for i, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        out = model(images)
        loss = criterion(out, labels)
        epoch_loss += loss.item()
        
        # Calculate accuracy
        _, y_true = torch.max(labels, 1)  #from one-hot encode to ordinal
        _, preds = torch.max(out, 1)  # Get the predicted class (with the highest score)
        correct_predictions += (preds == y_true).sum().item()  # Count correct predictions
        total_predictions += labels.size(0)  # Total number of labels in the batch
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"   step[{i+1} / {len(train_dataloader)}]")
    
    # Compute average loss and accuracy for the epoch
    epoch_loss /= len(train_dataloader)
    accuracy = correct_predictions / total_predictions * 100  # Convert to percentage
        
    return epoch_loss, accuracy
