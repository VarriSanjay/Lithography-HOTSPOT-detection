import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

def train_model():

    # 🔥 Device setup (GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ⚡ Improve performance
    torch.backends.cudnn.benchmark = True

    # 🎯 Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    # 📂 Dataset (make sure "train/" folder exists)
    train_dataset = datasets.ImageFolder("train", transform=train_transform)

    # 🚀 DataLoader (optimized for GPU)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,      # increased for GPU
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # 🧠 Model (ResNet18 pretrained)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Modify final layer for 2 classes
    model.fc = nn.Linear(model.fc.in_features, 2)

    model = model.to(device)

    # 🎯 Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 10

    # 🔁 Training loop
    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} Loss: {running_loss/len(train_loader):.4f}")

    # 💾 Save model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth")


# ✅ IMPORTANT: call function
if __name__ == "__main__":
    train_model()