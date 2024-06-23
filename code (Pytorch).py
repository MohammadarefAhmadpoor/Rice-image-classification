import os
from warnings import filterwarnings
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import splitfolders
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchinfo import summary
from sklearn.metrics import classification_report, confusion_matrix
filterwarnings('ignore')
path = '/kaggle/input/rice-image-dataset/Rice_Image_Dataset'
path = pathlib.Path(path)

splitfolders.ratio(path, output='df_splitted', seed=42, ratio=(0.7, 0.15, 0.15))

dir = '/kaggle/working/df_splitted'
dir = pathlib.Path(dir)

transformations = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std=1),
        transforms.Resize((250, 250))
    ]
)

Train = datasets.ImageFolder(os.path.join(dir, 'train'), transform = transformations)
train_loader = DataLoader(Train, batch_size=32, shuffle=True)
print(f'Train:\n {Train}\n\n')

Test = datasets.ImageFolder(os.path.join(dir, 'test'), transform = transformations)
test_loader = DataLoader(Test, batch_size=32, shuffle=True)
print(f'Test:\n {Test}\n\n')

Validation = datasets.ImageFolder(os.path.join(dir, 'val'), transform = transformations)
validation_loader = DataLoader(Validation, batch_size=32, shuffle=True)
print(f'Validation:\n {Validation}\n\n')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        
    def __iter__(self):
        for b in self.dataloader:
            yield to_device(b, self.device)
        
    def __len__(self):
        return len(self.dataloader)
    
    def switch_model_to_cpu(self, model):
        return model.to('cpu')
    
new_train_dataloader = DeviceDataLoader(train_loader, device)
new_validation_dataloader = DeviceDataLoader(validation_loader, device)

imgs, labels = next(iter(train_loader))
default_labels = {
    0: "Arborio",
    1: "Basmati",
    2: "Ipsala",
    3: "Jasmine",
    4: "Karacadag",
}
img_names = ['image_{}'.format(i) for i in range(len(imgs))]

fig, axes = plt.subplots(4, 8, figsize=(24, 12))
fig.suptitle('A batch of rice image samples', fontsize=20, fontweight='bold')

for ax, img, label_idx, img_name in zip(axes.flatten(), imgs, labels, img_names):
    img = torch.permute(img, (1, 2, 0))
    ax.imshow(img)
    ax.set_title(f'{default_labels[int(label_idx)]} - Tensor:{labels[int(label_idx)]}') 
    ax.axis('off')

plt.show()

class CNN(nn.Module):
    def __init__(self, In_channels, Num_classes):
        super(CNN, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=In_channels, out_channels=6, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.dense_layers = nn.Sequential(
            nn.Linear(59536, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, Num_classes)
        )

    def forward(self, x):
        op = self.cnn_layers(x)
        op = torch.flatten(op, 1)
        op = self.dense_layers(op)
        return op

model = to_device(CNN(3, len(Train.classes)), device)
summary(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
num_epochs = 5

train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(num_epochs):
    
    # Train
    model.train()
    train_loss = []
    correct_train = 0
    total_train = 0
    
    for images, labels in new_train_dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = np.mean(train_loss)
    train_acc = correct_train / total_train
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validation
    model.eval()
    val_loss = []
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for images, labels in new_validation_dataloader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss.append(loss.item())
            
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss = np.mean(val_loss)
    val_acc = correct_val / total_val
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f'Epoch {epoch+1}/{num_epochs} : Train Loss:{train_loss:.4f}, Train Acc:{train_acc:.4f}, Val Loss:{val_loss:.4f}, Val Acc:{val_acc:.4f}')

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, 'r--', label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, '--bo', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_accs, 'r--', label='Train Accuracy')
plt.plot(range(1, num_epochs+1), val_accs, '--bo',  label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()
plt.show()

new_test_dataloader = DeviceDataLoader(test_loader, device)

with torch.no_grad() :
    model.eval()
    test_loss = []
    correct_test = 0
    total_test = 0

    for images, labels in new_test_dataloader :

        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss.append(loss.item())

        _, prediction = torch.max(outputs, 1)
        total_test += labels.size(0)
        correct_test += (prediction==labels).sum().item()
        
    test_loss = np.mean(test_loss)
    test_acc = correct_test / total_test
    
    print(f'Test loss: {test_loss:.4f}\nTest Accuracy: {test_acc:.4f}')

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in new_test_dataloader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdGy')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

report = classification_report(all_labels, all_preds, target_names=Test.classes)
print("Classification Report:\n", report)

    