import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision
from PIL import Image
from train_epoch import *
from val_epoch import *


def train_model(num_epochs, model, train_dataloader, test_dataloader, optimizer, scheduler, criterion, device):
    best_score = float('inf')  # Initialize best score to a large number
    tol = 0  # Tolerance counter for early stopping
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}:")
        
        # Train the model
        model.train(True)
        avg_train_epoch_loss, train_accuracy = train_epoch(model, train_dataloader, device, criterion, optimizer)
        
        # Evaluate the model on validation data
        model.eval()
        avg_val_epoch_loss, val_accuracy = get_val_loss(model, test_dataloader, device, criterion)
        
        # Update the learning rate scheduler based on validation loss
        scheduler.step(avg_val_epoch_loss)
        
        # Save the best model based on validation loss
        if avg_val_epoch_loss < best_score:
            best_score = avg_val_epoch_loss
            torch.save(model.state_dict(), "best_model.pth")
            tol = 0
        else:
            tol += 1
            if tol == 5:
                print("Early stopping triggered.")
                break

        # Record the losses and accuracies
        train_losses.append(avg_train_epoch_loss)
        val_losses.append(avg_val_epoch_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch + 1} -- Avg Train Loss: {avg_train_epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Epoch {epoch + 1} -- Avg Val Loss: {avg_val_epoch_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        print("_" * 100)
    
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies
    }
