import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download

# -----------------------------
# 1️⃣ Load model function
# -----------------------------
def load_model(model_path: str, num_classes: int, device=None):
    """
    Load a ResNet18 model with custom number of classes.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model architecture
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return model

# -----------------------------
# 2️⃣ Download model from HF Hub
# -----------------------------
def get_model(num_classes=5, repo_id="tiennguyenbio/wbc-cnn-model", filename="wbc_model_gray.pth"):
    """
    Download model file from Hugging Face Hub and load it.
    """
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return load_model(model_path, num_classes)

# -----------------------------
# 3️⃣ Image preprocessing
# -----------------------------
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# -----------------------------
# 4️⃣ Labels dictionary
# -----------------------------
labels_dict = {
    0: "basophil",
    1: "eosinophil",
    2: "lymphocyte",
    3: "monocyte",
    4: "neutrophil"
}

# -----------------------------
# 5️⃣ Optional helper: predict PIL image
# -----------------------------
def predict_image(model, img: Image.Image, transform=test_transform, device=None):
    """
    Predict WBC class for a single PIL image.
    Returns: (class_index, class_label)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    img = img.convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        pred_index = torch.argmax(outputs, dim=1).item()
    
    return pred_index, labels_dict[pred_index]