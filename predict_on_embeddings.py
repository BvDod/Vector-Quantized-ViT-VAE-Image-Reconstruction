import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from models.cnn import CNNModel

    
def train_model(device, model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        running_loss = 0.0
        total = 0
        correct = 0
        
        ## Training
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
        print(f"Accuracy: {correct/total}")

        ## Validation
        model.eval()
        val_total = 0
        val_correct = 0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()
        val_accuracy = val_correct / val_total
        print(f"Validation Accuracy: {val_accuracy}\n")

if __name__ == "__main__":
    
    dir = "datasets/xray_embeddings/"

    # input_data = torch.load(dir + "quantized.pt")
    # input_data = torch.load(dir + "images_transformed.pt")
    input_data = torch.load(dir + "encoded.pt")
    labels = torch.load(dir + "labels.pt")
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a DataLoader
    dataset = TensorDataset(input_data, labels)

    # We already shuggled them before
    # Make train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize the model, criterion, and optimizer
    input_channels=input_data.shape[1]
    print(dataset[0][0].shape)
    model = CNNModel(input_channels=input_channels, shape=dataset[0][0].shape[1:]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1)

    # Train the model
    train_model(device, model, train_loader, val_loader, criterion, optimizer, num_epochs=25)