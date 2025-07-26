import torch
import torchvision.transforms as transforms
from PIL import Image
from models.tiny_cnn import Tiny_CNN

# 1. Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Tiny_CNN().to(device)
model.load_state_dict(torch.load("train_model.pth"))
model.eval()

# 2. Image Preprocessing Function
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),     # Convert to 1 channel
        transforms.Resize((28, 28)),                      # Resize to 28x28
        transforms.ToTensor(),                            # Convert to Tensor
        transforms.Normalize((0.1307,), (0.3081,))         # MNIST normalization
    ])
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension [1, 1, 28, 28]
    return image.to(device)

# 3. Prediction Function
def predict(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
        return predicted.item()

# 4. Main entry
if __name__ == "__main__":
    image_path = input("Enter image file path: ")
    result = predict(image_path)
    print(f"Predicted Digit: {result}")
