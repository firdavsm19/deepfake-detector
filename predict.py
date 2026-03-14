import sys
import torch
from PIL import Image
from dataset import val_test_transform
from model import DeepfakeDetector
from config import Config


def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepfakeDetector()
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval().to(device)

    image  = Image.open(image_path).convert("RGB")
    tensor = val_test_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output    = model(tensor)
        probs     = torch.softmax(output, dim=1)
        fake_prob = probs[0][1].item()
        real_prob = probs[0][0].item()

    label = "🔴 FAKE" if fake_prob > 0.5 else "🟢 REAL"
    print(f"\n  Image : {image_path}")
    print(f"  Result: {label}")
    print(f"  Real  : {real_prob:.2%}")
    print(f"  Fake  : {fake_prob:.2%}")
    return label, fake_prob


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"
    predict(image_path)