import gradio as gr
from model import get_model, test_transform, labels_dict
from PIL import Image
import torch

# Load model once
model = get_model()

# Prediction function
def classify_image(img: Image.Image):
    """
    img: PIL.Image uploaded via Gradio
    returns: string with label + confidence
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Ensure RGB
    img = img.convert("RGB")
    
    # Apply transform and batch dimension
    input_tensor = test_transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_index = torch.max(probs, 1)
    
    pred_label = labels_dict[pred_index.item()]
    return f"{pred_label} ({confidence.item()*100:.1f}%)"

# Gradio interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload Blood Cell Image"),
    outputs=gr.Textbox(label="Prediction"),
    title="White Blood Cell Classifier",
    description="Upload a blood cell image to classify it into one of five WBC types."
)

if __name__ == "__main__":
    iface.launch(share=True)