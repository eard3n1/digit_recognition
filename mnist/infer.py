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

def preprocess_image(img, target_size=28, digit_size=20):
    if isinstance(img, str):
        img = Image.open(img)

    img = img.convert("L")
    img = ImageOps.invert(img)
    bbox = img.getbbox()

    if bbox:
        img = img.crop(bbox)
        width, height = img.size
        scale = min(digit_size / width, digit_size / height)
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    else: img = img.resize((digit_size, digit_size), Image.Resampling.LANCZOS)

    new_img = Image.new("L", (target_size, target_size), 0)
    offset_x = (target_size - img.width) // 2
    offset_y = (target_size - img.height) // 2
    new_img.paste(img, (offset_x, offset_y))

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
    img = Image.open("./digit.png")
    digit, confidence = predict(model, img)
    print(f"Assumption: {digit} | Confidence: {confidence:.2f}%")
