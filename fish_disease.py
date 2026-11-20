import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "models/best_vit_model_90_target.pth"
IMAGE_PATH = "test_fish.jpeg"

device = torch.device("cpu")

# Load checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

class_names = checkpoint["class_names"]
image_size = checkpoint["image_size"]

print("Model:", checkpoint["model_name"])
print("Classes:", class_names)
print("Image size:", image_size)

# Create ViT-B/16
model = models.vit_b_16(
    weights=None,
    num_classes=len(class_names)
)

# Load trained weights
state_dict = checkpoint["model_state_dict"]

# Remove DataParallel prefix if present
new_state_dict = {}

for key, value in state_dict.items():
    if key.startswith("module."):
        key = key[7:]
    new_state_dict[key] = value

model.load_state_dict(new_state_dict)

model = model.to(device)
model.eval()

print("ViT model loaded successfully.")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load image
image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# Prediction
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

predicted_class = class_names[predicted.item()]
confidence_value = confidence.item() * 100

print()
print("=" * 40)
print("FISH DISEASE PREDICTION")
print("=" * 40)
print("Prediction :", predicted_class)
print("Confidence :", f"{confidence_value:.2f}%")
print("=" * 40)

print()
print("All class probabilities:")

for name, probability in zip(class_names, probabilities[0]):
    print(f"{name:12s}: {probability.item() * 100:.2f}%")
