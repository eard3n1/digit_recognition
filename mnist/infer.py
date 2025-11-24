import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
try: from .model import MNIST
except ImportError: from model import MNIST

device = torch.device("cpu") # Inference dosen't neccessarily require GPU

def load_model(path="./models/mnist_cnn.pth"):
    model = MNIST()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(img):
    img = img.convert("L")
    img = ImageOps.invert(img)
    bbox = img.getbbox()
    if bbox: img = img.crop(bbox)
    img = img.resize((20, 20), Image.Resampling.LANCZOS)
    new_img = Image.new("L", (28, 28), 0)
    new_img.paste(img, ((28-20)//2, (28-20)//2))
    new_img = new_img.filter(ImageFilter.GaussianBlur(1))
    transform = transforms.ToTensor()
    return transform(new_img).unsqueeze(0).to(device)

def predict(model, img):
    tensor = preprocess_image(img)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)
    return pred.item(), confidence.item() * 100

if __name__ == "__main__":
    model = load_model()
    img = Image.open("digit.png")
    digit, confidence = predict(model, img)
    print(f"Assumption: {digit} | Confidence: {confidence:.2f}%")
