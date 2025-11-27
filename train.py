import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mnist.model import MNIST
import matplotlib.pyplot as plt
import os

def main():
    batch_size = 64
    epochs = 5
    lr = 0.001
    path = "models/mnist_cnn.pth"

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        transform=transform,
        download=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cpu") # Could use "cuda", but unnecessary for small-scale
    model = MNIST().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    acc_history = []

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total * 100

        loss_history.append(avg_loss)
        acc_history.append(accuracy)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f'Model saved: "{path}"')

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Loss Graph
    ax1.plot(range(1, epochs+1), loss_history, 'o-', color='red', label='Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True)
    ax1.set_xticks(range(1, epochs+1))
    ax1.set_ylim(min(loss_history) * 0.95, max(loss_history) * 1.05)

    # Accuracy Graph
    ax2 = ax1.twinx()
    ax2.plot(range(1, epochs+1), acc_history, 'o-', color='blue', label='Accuracy')
    ax2.set_ylabel('Accuracy (%)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim(min(acc_history) * 0.95, max(acc_history) * 1.05)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
  
    plt.title("Loss & Accuracy")
    plt.show()

if __name__ == "__main__":
    main()
