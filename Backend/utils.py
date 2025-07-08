from torchvision import transforms
from PIL import Image
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_file):
    image = Image.open(image_file).convert('RGB')
    image = transform(image).unsqueeze(0)  # Shape: [1, 3, 224, 224]
    return image.to(DEVICE)
