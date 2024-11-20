import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import json
import os
from tqdm import tqdm

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# Custom CNN model
class CustomCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(CustomCNN, self).__init__()
        # Conv Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Conv Block 3
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 256)    # 28x28 is feature map size due to original being 224x224 and there are 3 pooling layers
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Conv Block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Conv Block 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        # Conv Block 3
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx]['image_path']
        label = self.dataframe.iloc[idx]['label']

        # Check if the image path exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path {image_path} does not exist.")

        # Load the image
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Convert label to a tensor
        label_tensor = torch.tensor(ord(label) - ord('A')).long()  # Map 'A' to 0, 'B' to 1, etc.

        return image, label_tensor

# Load data splits from CSV
data_splits_csv = '/home/UNT/jtc0129/Desktop/continualLearning/data_splits.csv'
task_A_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
data_splits = pd.read_csv(data_splits_csv)
task_A_splits = data_splits[data_splits['label'].isin(task_A_classes)].reset_index(drop=True)

# Print the number of entries in each split
print("Number of training samples:", len(task_A_splits[task_A_splits['split'] == 'train']))
print("Number of validation samples:", len(task_A_splits[task_A_splits['split'] == 'validation']))
print("Number of test samples:", len(task_A_splits[task_A_splits['split'] == 'test']))


# Define parameters for grid search
learning_rates = [0.001, 0.01, 0.1]
dropout_rates = [0.4, 0.5, 0.6]
optimizers = ['SGD', 'Adam']
batch_size = 32
num_epochs = 100
early_stopping_patience = 5

# Set file paths
log_file_path = '/home/UNT/jtc0129/Desktop/continualLearning/grid_search_results.json'
model_save_dir = '/home/UNT/jtc0129/Desktop/continualLearning/saved_models'
os.makedirs(model_save_dir, exist_ok=True)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Function to train and validate the model
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, early_stopping_patience, model_save_path):
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        val_accuracy = correct / total if total > 0 else 0
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

    return history

# Grid search function
def grid_search(device):
    results = []
    for lr in learning_rates:
        for dropout_rate in dropout_rates:
            for opt in optimizers:
                print(f"Testing: LR={lr}, Dropout={dropout_rate}, Optimizer={opt}")
                model = CustomCNN(num_classes=8)
                model.dropout = nn.Dropout(dropout_rate)
                optimizer = optim.SGD(model.parameters(), lr=lr) if opt == 'SGD' else optim.Adam(model.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss()

                train_data = task_A_splits[task_A_splits['split'] == 'train']
                val_data = task_A_splits[task_A_splits['split'] == 'validation']

                train_dataset = CustomDataset(train_data, transform=transform)
                val_dataset = CustomDataset(val_data, transform=transform)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                model_save_path = os.path.join(model_save_dir, f'best_model_lr{lr}_dropout{dropout_rate}_opt{opt}.pth')
                history = train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, early_stopping_patience, model_save_path)

                result = {
                    'learning_rate': lr,
                    'dropout_rate': dropout_rate,
                    'optimizer': opt,
                    'train_loss': history['train_loss'],
                    'val_loss': history['val_loss'],
                    'val_accuracy': history['val_accuracy']
                }
                results.append(result)
                with open(log_file_path, 'w') as f:
                    json.dump(results, f, indent=4)

                del model
                del optimizer
                torch.cuda.empty_cache()

# Main script
if __name__ == "__main__":
    grid_search(device)