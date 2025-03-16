import os
import zipfile
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import timm

# Extracting the dataset from the zipfile
zip_path = r"C:\Users\abbas\OneDrive\Desktop\Hackathon\FER2013.zip"
extract_folder = r"C:\Users\abbas\OneDrive\Desktop\Hackathon\FER2013_dataset"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

print(f"Dataset extracted to {extract_folder}")


# Preprocessing the dataset suitable for Swin Transformer
train_dir = r"C:\Users\abbas\OneDrive\Desktop\Hackathon\FER2013_dataset\train"

# 📌 Define transformations (Convert grayscale to RGB)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel RGB
    transforms.Resize((224, 224)),  # Resize for Swin Transformer
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize images
])
#
# 📌 Load datasets
train_dataset = ImageFolder(root=train_dir, transform=transform)

# 📌 Create Data Loaders
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=0, pin_memory=True)

print(f"Dataset loaded successfully! Training samples: {len(train_dataset)}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("✅ Checking GPU availability...")
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
print("Current CUDA Device:", torch.cuda.current_device() if torch.cuda.is_available() else "No GPU detected")
print("CUDA Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=7)
model.to(device)

print(model)


# Setting up Loss function and optimizer
import torch.optim as optim

# 📌 Define Loss Function (Cross Entropy for Multi-Class Classification)
criterion = nn.CrossEntropyLoss()

# 📌 Define Optimizer (AdamW works well for Transformers)
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)

print("✅ Loss function and optimizer initialized!")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)  # Ensures proper multiprocessing handling

"""Fine-Tuning the model"""
from tqdm import tqdm  # For progress bar
import torch.cuda.amp as amp  # For Mixed Precision Training

# 📌 Define number of epochs
num_epochs = 5
accumulation_steps = 4  # Gradient Accumulation (useful for small batch sizes)

# 📌 Enable Mixed Precision Training
scaler = torch.amp.GradScaler()

# 📌 Training Loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

    for i, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # Reset gradients

        with amp.autocast():  # Enable FP16 computation
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps  # Scale loss for gradient accumulation

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:  # Update weights after `accumulation_steps` batches
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss / (i + 1))

    print(f"✅ Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

print("🎉 Training Complete! Saving model...")

# 📌 Save Model
torch.save(model.state_dict(), "swin_tiny_fer2013.pth")
print("✅ Model saved as 'swin_tiny_fer2013.pth' 🚀")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
