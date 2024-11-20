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

# List of letters excluding 'J' and 'Z'
letters = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
    'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
    'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
]

# Create a mapping from letters to indices
letter_to_index = {letter: idx for idx, letter in enumerate(letters)}
index_to_letter = {idx: letter for idx, letter in enumerate(letters)}

# Custom CNN model
class CustomCNN(nn.Module):
    def __init__(self, num_classes=24):
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
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.dropout = nn.Dropout(0.6)  # Using dropout rate from the best model
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
        self.dataframe = dataframe.reset_index(drop=True)
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

        # Convert label to a tensor using the custom mapping
        label_index = letter_to_index[label]
        label_tensor = torch.tensor(label_index).long()

        return image, label_tensor

# Load data splits from CSV
data_splits_csv = '/home/UNT/jtc0129/Desktop/continualLearning/data_splits.csv'
data_splits = pd.read_csv(data_splits_csv)

# Define classes for each task
task_A_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
task_B_classes = ['I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q']
task_C_classes = ['R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Prepare data splits for each task
task_A_splits = data_splits[data_splits['label'].isin(task_A_classes)].reset_index(drop=True)
task_B_splits = data_splits[data_splits['label'].isin(task_B_classes)].reset_index(drop=True)
task_C_splits = data_splits[data_splits['label'].isin(task_C_classes)].reset_index(drop=True)

# Define parameters
learning_rate = 0.01    # From the best model
dropout_rate = 0.6      # From the best model
optimizer_type = 'SGD'  # From the best model
batch_size = 32
num_epochs_per_task = 100
early_stopping_patience = 5

# Set file paths
model_save_dir = '/home/UNT/jtc0129/Desktop/continualLearning/saved_models/baseline_forgetting'
os.makedirs(model_save_dir, exist_ok=True)
model_save_path = os.path.join(model_save_dir, 'baseline_forgetting_model.pth')
log_file_path = '/home/UNT/jtc0129/Desktop/continualLearning/metrics/forget/baseline_forgetting_results.json'

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Prepare datasets and dataloaders for each task
def prepare_dataloaders(task_splits):
    train_data = task_splits[task_splits['split'] == 'train']
    val_data = task_splits[task_splits['split'] == 'validation']
    test_data = task_splits[task_splits['split'] == 'test']

    train_dataset = CustomDataset(train_data, transform=transform)
    val_dataset = CustomDataset(val_data, transform=transform)
    test_dataset = CustomDataset(test_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

train_loader_A, val_loader_A, test_loader_A = prepare_dataloaders(task_A_splits)
train_loader_B, val_loader_B, test_loader_B = prepare_dataloaders(task_B_splits)
train_loader_C, val_loader_C, test_loader_C = prepare_dataloaders(task_C_splits)

# Indices and label mappings for each task
task_A_indices = [letter_to_index[letter] for letter in task_A_classes]
task_B_indices = [letter_to_index[letter] for letter in task_B_classes]
task_C_indices = [letter_to_index[letter] for letter in task_C_classes]

label_mapping_A = {letter_to_index[letter]: idx for idx, letter in enumerate(task_A_classes)}
label_mapping_B = {letter_to_index[letter]: idx for idx, letter in enumerate(task_B_classes)}
label_mapping_C = {letter_to_index[letter]: idx for idx, letter in enumerate(task_C_classes)}

# Initialize the model with 24 output classes
num_classes_total = 24
model = CustomCNN(num_classes=num_classes_total)
model.dropout = nn.Dropout(dropout_rate)  # Set dropout rate

# Move the model to the device
model.to(device)

# Define the optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Initialize history dictionary
history = {
    'epoch': [],
    'train_loss': [],
    'val_loss': [],
    'val_accuracy': [],
    'test_A_accuracy': [],
    'test_B_accuracy': [],
    'test_C_accuracy': [],
    'tasks': []  # To keep track of task start and end epochs
}

# Training function
def train_on_task(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, early_stopping_patience, model_save_path, current_epoch, task_name, task_indices, label_mapping, history):
    best_val_loss = float('inf')
    patience_counter = 0
    start_epoch = current_epoch + 1  # For logging task start epoch
    best_epoch = current_epoch
    best_model_state_dict = None

    for epoch in range(num_epochs):
        global_epoch = current_epoch + epoch + 1
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {global_epoch} - Training on {task_name}'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            # Get outputs for current task classes
            outputs_task = outputs[:, task_indices]

            # Map labels to 0-(num_classes-1) for current task
            labels_task = torch.tensor([label_mapping[label.item()] for label in labels]).to(device)

            loss = criterion(outputs_task, labels_task)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {global_epoch} - Validation on {task_name}'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # Get outputs for current task classes
                outputs_task = outputs[:, task_indices]
                labels_task = torch.tensor([label_mapping[label.item()] for label in labels]).to(device)

                loss = criterion(outputs_task, labels_task)
                val_loss += loss.item()
                _, predicted = torch.max(outputs_task, 1)
                total_val += labels_task.size(0)
                correct_val += (predicted == labels_task).sum().item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        val_accuracy = correct_val / total_val if total_val > 0 else 0

        # Evaluate on all tasks
        test_accuracies = {}
        for task in ['A', 'B', 'C']:
            test_loader = globals()[f'test_loader_{task}']
            task_indices_eval = globals()[f'task_{task}_indices']
            label_mapping_eval = globals()[f'label_mapping_{task}']
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)

                    # Get outputs for the task classes
                    outputs_task_eval = outputs[:, task_indices_eval]
                    labels_task_eval = torch.tensor([label_mapping_eval.get(label.item(), -1) for label in labels]).to(device)

                    # Filter out labels not in the current mapping
                    valid_indices = labels_task_eval != -1
                    if valid_indices.sum().item() == 0:
                        continue
                    outputs_task_eval = outputs_task_eval[valid_indices]
                    labels_task_eval = labels_task_eval[valid_indices]

                    _, predicted = torch.max(outputs_task_eval, 1)
                    total_test += labels_task_eval.size(0)
                    correct_test += (predicted == labels_task_eval).sum().item()

            test_accuracy = correct_test / total_test if total_test > 0 else 0
            test_accuracies[f'test_{task}_accuracy'] = test_accuracy

        # Log metrics
        history['epoch'].append(global_epoch)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['test_A_accuracy'].append(test_accuracies['test_A_accuracy'])
        history['test_B_accuracy'].append(test_accuracies['test_B_accuracy'])
        history['test_C_accuracy'].append(test_accuracies['test_C_accuracy'])

        print(f'Epoch [{global_epoch}]')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Accuracy on {task_name}: {val_accuracy:.4f}')
        print(f"Test Accuracy on Task A: {test_accuracies['test_A_accuracy']:.4f}")
        print(f"Test Accuracy on Task B: {test_accuracies['test_B_accuracy']:.4f}")
        print(f"Test Accuracy on Task C: {test_accuracies['test_C_accuracy']:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = global_epoch
            best_model_state_dict = model.state_dict()
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

        # Save metrics after each epoch
        with open(log_file_path, 'w') as f:
            json.dump(history, f, indent=4)

    actual_end_epoch = global_epoch

    # Load the best model
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
    else:
        # If no improvement was found during training
        best_epoch = actual_end_epoch

    # Adjust history to only include up to best_epoch
    best_epoch_index = history['epoch'].index(best_epoch)
    for key in ['epoch', 'train_loss', 'val_loss', 'val_accuracy', 'test_A_accuracy', 'test_B_accuracy', 'test_C_accuracy']:
        history[key] = history[key][:best_epoch_index + 1]

    # Update current_epoch to best_epoch
    current_epoch = best_epoch

    # Record task start and end epochs
    history['tasks'].append({
        'task_name': task_name,
        'start_epoch': start_epoch,
        'end_epoch': best_epoch,
        'actual_end_epoch': actual_end_epoch
    })

    # Save adjusted metrics
    with open(log_file_path, 'w') as f:
        json.dump(history, f, indent=4)

    return current_epoch


# Main script
if __name__ == "__main__":
    current_epoch = 0  # Start from epoch 0

    # Training on Task A
    current_epoch = train_on_task(
        model,
        train_loader_A,
        val_loader_A,
        criterion,
        optimizer,
        device,
        num_epochs_per_task,
        early_stopping_patience,
        model_save_path,
        current_epoch,
        'Task A',
        task_A_indices,
        label_mapping_A,
        history
    )

    # Load the best model after Task A
    model.load_state_dict(torch.load(model_save_path, map_location=device))

    # Training on Task B
    current_epoch = train_on_task(
        model,
        train_loader_B,
        val_loader_B,
        criterion,
        optimizer,
        device,
        num_epochs_per_task,
        early_stopping_patience,
        model_save_path,
        current_epoch,
        'Task B',
        task_B_indices,
        label_mapping_B,
        history
    )

    # Load the best model after Task B
    model.load_state_dict(torch.load(model_save_path, map_location=device))

    # Training on Task C
    current_epoch = train_on_task(
        model,
        train_loader_C,
        val_loader_C,
        criterion,
        optimizer,
        device,
        num_epochs_per_task,
        early_stopping_patience,
        model_save_path,
        current_epoch,
        'Task C',
        task_C_indices,
        label_mapping_C,
        history
    )

    # Load the best model after Task C
    model.load_state_dict(torch.load(model_save_path, map_location=device))

    # Final evaluation on test sets
    model.eval()
    final_test_accuracies = {}
    for task in ['A', 'B', 'C']:
        test_loader = globals()[f'test_loader_{task}']
        task_indices_eval = globals()[f'task_{task}_indices']
        label_mapping_eval = globals()[f'label_mapping_{task}']
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # Get outputs for the task classes
                outputs_task_eval = outputs[:, task_indices_eval]
                labels_task_eval = torch.tensor([label_mapping_eval.get(label.item(), -1) for label in labels]).to(device)

                # Filter out labels not in the current mapping
                valid_indices = labels_task_eval != -1
                if valid_indices.sum().item() == 0:
                    continue
                outputs_task_eval = outputs_task_eval[valid_indices]
                labels_task_eval = labels_task_eval[valid_indices]

                _, predicted = torch.max(outputs_task_eval, 1)
                total_test += labels_task_eval.size(0)
                correct_test += (predicted == labels_task_eval).sum().item()

        test_accuracy = correct_test / total_test if total_test > 0 else 0
        final_test_accuracies[f'test_{task}_accuracy'] = test_accuracy
        print(f"Final Test Accuracy on Task {task}: {test_accuracy:.4f}")

    # Save the final test accuracies
    history['final_test_accuracies'] = final_test_accuracies
    with open(log_file_path, 'w') as f:
        json.dump(history, f, indent=4)
