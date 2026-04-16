import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18

def test_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    test_dataset = datasets.ImageFolder("test", transform=test_transform)

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = resnet18()

    model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(torch.load("model.pth"))

    model = model.to(device)

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print("Test Accuracy:", accuracy)