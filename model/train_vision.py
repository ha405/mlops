import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
import argparse
import os

def train_model(epochs=1, fast_dev_run=False, save_path='vision_model.pth'):
    print(f"Setting up MobileNetV2 for finetuning on CIFAR10 (epochs={epochs}, fast_dev_run={fast_dev_run})...")
    
    # Use CIFAR10 dataset
    transform = transforms.Compose([
        transforms.Resize(224), # MobileNetV2 expects 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Download dataset
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # If fast_dev_run, just use a tiny subset of the data
    if fast_dev_run:
        trainset = torch.utils.data.Subset(trainset, range(100))
        
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    
    # Load pre-trained model
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Freeze earlier layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace classifier for 10 classes (CIFAR10)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 10)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Optimize only the classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 10 == 9:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0
                
    print('Finished Fine-tuning')
    
    # Save the model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    # Save class labels mapping
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    class_path = os.path.join(os.path.dirname(__file__), 'cifar10_classes.txt')
    with open(class_path, 'w') as f:
        f.write('\n'.join(classes))
    print(f"Saved class labels to {class_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Vision Model')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--fast', action='store_true', help='Run a fast dev run (few batches)')
    args = parser.parse_args()
    
    # Always save to model/ directory relative to root
    save_path = os.path.join(os.path.dirname(__file__), 'vision_model.pth')
    train_model(epochs=args.epochs, fast_dev_run=args.fast, save_path=save_path)
