import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# 📌 Load the trained model
model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=7)
model.load_state_dict(torch.load("swin_tiny_fer2013.pth", map_location=device))  # Load saved model
model.to(device)  # Move model to GPU/CPU
model.eval()  # Set model to evaluation mode

print("✅ Model loaded successfully!")

# # Transformation
transform = transforms.Compose([transforms.Resize((224, 224)),  # Resize for Swin Transformer
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize
])
# 📌 Load and preprocess the test image
image_path = "C:/Users/abbas/OneDrive/Desktop/Hackathon/test_image.jpg"  # 🔹 Replace this with your image path
image = Image.open(image_path)
image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to GPU/CPU

# Inference
with torch.no_grad():
    outputs = model(image)
    predicted_class = torch.argmax(outputs, dim=1).item()  # Get the predicted class index

# 📌 Define emotion labels (FER2013 classes)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# 📌 Print the predicted emotion
predicted_emotion = emotion_labels[predicted_class]
print(f"✅ Predicted Emotion: {predicted_emotion}")

#
# #Pre-process the test
# test_dir = r"C:/Users/abbas/OneDrive/Desktop/Hackathon/FER2013_dataset/test"
#
# transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
#     transforms.Resize((224, 224)),  # Resize for Swin Transformer
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])  # Normalize images
# ])
# test_dataset = ImageFolder(root=test_dir, transform=transform)

#
# # 📌 Select 25 random images from the test dataset
# subset_size = 25
# indices = np.random.choice(len(test_dataset), subset_size, replace=False)
# test_subset = Subset(test_dataset, indices)
#
#
# # Create a DataLoader for batch processing
# test_loader = DataLoader(test_subset, batch_size=5, shuffle=False, num_workers=0, pin_memory=True)
# print(f"✅ Full test dataset loaded: {len(test_loader.dataset)} samples")
#
#
# #
# from sklearn.metrics import accuracy_score, classification_report
#
# # 📌 Define emotion labels for FER2013
# emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
#
# # 📌 Initialize lists to store predictions and actual labels
# all_preds = []
# all_labels = []
#
# # 📌 Set model to evaluation mode
# model.eval()
# with torch.no_grad():
#     for images, labels in test_loader:
#         images, labels = images.to(device), labels.to(device)
#
#         outputs = model(images)
#         _, preds = torch.max(outputs, 1)
#
#         all_preds.extend(preds.cpu().numpy())  # Move to CPU for sklearn processing
#         all_labels.extend(labels.cpu().numpy())
#
# # 📌 Calculate accuracy
# accuracy = accuracy_score(all_labels, all_preds)
# print(f"✅ Model Accuracy on Full Test Set: {accuracy:.4f}")
#
# # 📌 Generate a classification report
# print("📊 Classification Report:\n", classification_report(all_labels, all_preds, target_names=emotion_labels))
