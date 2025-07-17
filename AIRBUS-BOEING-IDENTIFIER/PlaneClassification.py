import torch
from torchvision import models, transforms
from PIL import Image
import sys
import torch.nn as nn


model_path = "aircraft_classifier_cpu.pth"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if len(sys.argv) != 2:
    print("Usage: python3 classify_any.py path/to/image.jpg")
    sys.exit(1)

image_path = sys.argv[1]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


try:
    image = Image.open(image_path).convert("RGB")
except Exception as e:
    print(f"❌ Error loading image: {e}")
    sys.exit(1)

image_tensor = transform(image).unsqueeze(0).to(device)  


checkpoint = torch.load(model_path, map_location=device)
class_names = checkpoint["class_names"]
num_classes = len(class_names)


model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()


with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    confidence, predicted_idx = torch.max(probabilities, 1)
    predicted_label = class_names[predicted_idx.item()]
    confidence_pct = confidence.item() * 100

print(f"Aircraft Model: {predicted_label} ({confidence_pct:.2f}% confidence)")