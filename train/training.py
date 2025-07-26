import torch
import torch.nn as nn
import torch.optim as optim
from models.tiny_cnn import Tiny_CNN
from data.mnist_loader import load_mnist

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        a,predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100 * correct / total
    print(f"Train Loss: {running_loss:.4f} | Train Accuracy: {acc:.2f}%")

def test(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            a, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Tiny_CNN().to(device)
    train_loader, test_loader = load_mnist(batch_size=16)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        print(f"Epoch {epoch+1}")
        train(model, train_loader, optimizer, criterion, device)
    torch.save(model.state_dict(), "train_model.pth")
    test(model, test_loader, device)
