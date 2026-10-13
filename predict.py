import sys
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import timm
import pandas as pd

class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super(SimpleCardClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=False)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output

def get_class_labels(csv_path):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        class_mapping = df.drop_duplicates(subset=['class index']).sort_values('class index')
        return dict(zip(class_mapping['class index'], class_mapping['labels']))
    return {}

def predict(image_path, model_path='simple_card_classifier.pt', csv_path='Card Image Dataset-Classification/cards.csv'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCardClassifier(num_classes=53).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image '{image_path}' not found.")

    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
        probs = torch.softmax(output, dim=1)[0]
        confidence = probs[predicted_idx].item()

    labels = get_class_labels(csv_path)
    label_name = labels.get(predicted_idx, f"Class {predicted_idx}")

    print(f"[*] Input Image : {image_path}")
    print(f"[*] Prediction  : {label_name.title()} (Index: {predicted_idx})")
    print(f"[*] Confidence  : {confidence * 100:.2f}%")
    return label_name

if __name__ == '__main__':
    default_img = os.path.join('Card Image Dataset-Classification', 'test', 'ace of clubs', '1.jpg')
    img_path = sys.argv[1] if len(sys.argv) > 1 else default_img
    predict(img_path)
