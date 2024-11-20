import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import json
import os
from tqdm import tqdm
import copy  # Import copy module for deep copies

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# List of letters excluding 'J' and 'Z'
letters = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',  # Task A classes
    'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',  # Task B classes
    'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'   # Task C classes
]

# Create a mapping from letters to indices
letter_to_index = {letter: idx for idx, letter in enumerate(letters)}
index_to_letter = {idx: letter for idx, letter in enumerate(letters)}

# Define Task A, Task B, and Task C classes
task_A_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
task_B_classes = ['I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q']
task_C_classes = ['R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Indices and label mappings for each task
task_indices = {
    'A': [letter_to_index[letter] for letter in task_A_classes],
    'B': [letter_to_index[letter] for letter in task_B_classes],
    'C': [letter_to_index[letter] for letter in task_C_classes]
}

label_mappings = {
    'A': {letter_to_index[letter]: idx for idx, letter in enumerate(task_A_classes)},
    'B': {letter_to_index[letter]: idx for idx, letter in enumerate(task_B_classes)},
    'C': {letter_to_index[letter]: idx for idx, letter in enumerate(task_C_classes)}
}

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
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        # Conv Block 2
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool(x)

        # Conv Block 3
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
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

        # Determine the task
        if label in task_A_classes:
            task = 'A'
        elif label in task_B_classes:
            task = 'B'
        else:
            task = 'C'

        return image, label_tensor, task

# Load data splits from CSV
data_splits_csv = '/home/UNT/jtc0129/Desktop/continualLearning/data_splits.csv'
data_splits = pd.read_csv(data_splits_csv)

# Filter data for each task
task_splits = {}
for task, classes in zip(['A', 'B', 'C'], [task_A_classes, task_B_classes, task_C_classes]):
    task_splits[task] = data_splits[data_splits['label'].isin(classes)].reset_index(drop=True)

# Print the number of entries in each split for each task
for task in ['A', 'B', 'C']:
    print(f"Task {task} - Number of training samples:", len(task_splits[task][task_splits[task]['split'] == 'train']))
    print(f"Task {task} - Number of validation samples:", len(task_splits[task][task_splits[task]['split'] == 'validation']))
    print(f"Task {task} - Number of test samples:", len(task_splits[task][task_splits[task]['split'] == 'test']))

# Define parameters
learning_rate = 0.01    # From the best model
dropout_rate = 0.6      # From the best model
optimizer_type = 'SGD'  # From the best model
batch_size = 32
num_epochs = 100
early_stopping_patience = 5

# Set file paths
model_save_dir = '/home/UNT/jtc0129/Desktop/continualLearning/saved_models/replay'
os.makedirs(model_save_dir, exist_ok=True)
metrics_file_path = '/home/UNT/jtc0129/Desktop/continualLearning/metrics/replay_metrics.json'

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Initialize model with 24 output classes
num_classes_total = 24

# Training and validation function without modifying global history
def train_and_validate(model, train_loader, val_loaders, criterion, optimizer, device, num_epochs, early_stopping_patience, model_save_path, current_epoch, task_name):
    best_val_loss = float('inf')
    patience_counter = 0
    start_epoch = current_epoch + 1  # For logging task start epoch
    best_epoch = current_epoch
    best_model_state_dict = None

    # Initialize local task history
    task_history = {
        'epoch': [],
        'train_loss': [],
    }
    for task in ['A', 'B', 'C']:
        task_history[f'val_loss_{task}'] = []
        task_history[f'val_accuracy_{task}'] = []

    for epoch in range(num_epochs):
        global_epoch = current_epoch + epoch + 1
        model.train()
        train_loss = 0.0
        for inputs, labels, tasks in tqdm(train_loader, desc=f'Epoch {global_epoch} - Training on {task_name}'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass through model
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation on all tasks
        model.eval()
        val_losses = {}
        val_accuracies = {}
        total_val_loss = 0.0

        with torch.no_grad():
            for task in ['A', 'B', 'C']:
                val_loader = val_loaders[task]
                val_loss = 0.0
                correct = 0
                total = 0
                for inputs, labels, tasks in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    outputs_task = outputs[:, task_indices[task]]
                    labels_mapped = torch.tensor([label_mappings[task][label.item()] for label in labels], device=device)

                    loss = criterion(outputs_task, labels_mapped)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs_task, 1)
                    total += labels_mapped.size(0)
                    correct += (predicted == labels_mapped).sum().item()

                avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
                accuracy = correct / total if total > 0 else 0

                val_losses[task] = avg_val_loss
                val_accuracies[task] = accuracy
                total_val_loss += avg_val_loss

                # Log history
                task_history[f'val_loss_{task}'].append(avg_val_loss)
                task_history[f'val_accuracy_{task}'].append(accuracy)

        # Average validation loss across tasks
        avg_total_val_loss = total_val_loss / len(val_loaders)

        task_history['epoch'].append(global_epoch)
        task_history['train_loss'].append(avg_train_loss)

        print(f'Epoch [{global_epoch}]')
        print(f'Train Loss: {avg_train_loss:.4f}')
        for task in ['A', 'B', 'C']:
            print(f'Val Loss on Task {task}: {val_losses[task]:.4f}, Val Accuracy on Task {task}: {val_accuracies[task]:.4f}')

        # Early stopping based on average validation loss
        if avg_total_val_loss < best_val_loss:
            best_val_loss = avg_total_val_loss
            patience_counter = 0
            best_epoch = global_epoch
            best_model_state_dict = copy.deepcopy(model.state_dict())
            # Save the best model
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

    actual_end_epoch = global_epoch

    # Load the best model
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
    else:
        # If no improvement was found during training
        best_epoch = actual_end_epoch

    # Adjust task_history to include only up to best_epoch
    best_epoch_index = task_history['epoch'].index(best_epoch)
    for key in task_history.keys():
        task_history[key] = task_history[key][:best_epoch_index + 1]

    # Record task start and end epochs
    task_history['tasks'] = [{
        'task_name': task_name,
        'start_epoch': start_epoch,
        'end_epoch': best_epoch,
        'actual_end_epoch': actual_end_epoch
    }]

    return task_history, best_epoch

# Evaluation function
def evaluate_model(model, test_loaders, criterion, device):
    model.eval()
    test_accuracies = {}
    with torch.no_grad():
        for task in ['A', 'B', 'C']:
            test_loader = test_loaders[task]
            correct = 0
            total = 0
            for inputs, labels, tasks in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                outputs_task = outputs[:, task_indices[task]]
                labels_mapped = torch.tensor([label_mappings[task][label.item()] for label in labels], device=device)

                _, predicted = torch.max(outputs_task, 1)
                total += labels_mapped.size(0)
                correct += (predicted == labels_mapped).sum().item()

            accuracy = correct / total if total > 0 else 0
            test_accuracies[task] = accuracy
            print(f'Test Accuracy on Task {task}: {accuracy:.4f}')

    return test_accuracies

# Main script
if __name__ == "__main__":
    # Define replay percentages to test
    replay_percentages = [0.05, 0.1, 0.15, 0.2]  # 5%, 10%, 15%, 20%

    # Prepare validation and test loaders for all tasks
    val_loaders = {}
    test_loaders = {}
    for task in ['A', 'B', 'C']:
        val_data = task_splits[task][task_splits[task]['split'] == 'validation']
        val_dataset = CustomDataset(val_data, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        val_loaders[task] = val_loader

        test_data = task_splits[task][task_splits[task]['split'] == 'test']
        test_dataset = CustomDataset(test_data, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_loaders[task] = test_loader

    # Initialize a dictionary to store all metrics
    all_metrics = {}

    # Step 1: Train on Task A
    print("\nTraining on Task A")
    # Training data for Task A
    train_data_A = task_splits['A'][task_splits['A']['split'] == 'train']
    train_dataset_A = CustomDataset(train_data_A, transform=transform)
    train_loader_A = DataLoader(train_dataset_A, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = CustomCNN(num_classes=num_classes_total)
    model.dropout = nn.Dropout(dropout_rate)  # Set dropout rate
    model.to(device)

    # Define optimizer and criterion
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Set model save path
    model_save_path_A = os.path.join(model_save_dir, 'model_task_A.pth')

    current_epoch = 0  # Start from epoch 0

    # Train and validate on Task A
    task_history_A, best_epoch_A = train_and_validate(
        model,
        train_loader_A,
        val_loaders,
        criterion,
        optimizer,
        device,
        num_epochs,
        early_stopping_patience,
        model_save_path_A,
        current_epoch,
        'Task A'
    )

    # Store Task A history separately
    all_metrics['Task_A'] = {
        'training_history': copy.deepcopy(task_history_A),
        'test_accuracies': None  # Will be updated after evaluation
    }
    task_A_history = copy.deepcopy(task_history_A)

    # Update current_epoch
    current_epoch = best_epoch_A

    # Evaluate on test sets
    print("\nEvaluating on test sets after training on Task A")
    model.load_state_dict(torch.load(model_save_path_A))
    test_accuracies_A = evaluate_model(model, test_loaders, criterion, device)

    # Update metrics
    all_metrics['Task_A']['test_accuracies'] = test_accuracies_A

    # Release GPU memory
    del model
    del optimizer
    torch.cuda.empty_cache()

    # Step 2: Train on Task B and Task C with Replay
    for replay_frac in replay_percentages:
        print(f"\nTraining with replay fraction: {replay_frac*100}%")

        # Initialize cumulative history for this replay percentage
        cumulative_history = {
            'epoch': task_A_history['epoch'].copy(),
            'train_loss': task_A_history['train_loss'].copy(),
        }
        for task in ['A', 'B', 'C']:
            cumulative_history[f'val_loss_{task}'] = task_A_history.get(f'val_loss_{task}', []).copy()
            cumulative_history[f'val_accuracy_{task}'] = task_A_history.get(f'val_accuracy_{task}', []).copy()
        cumulative_history['tasks'] = task_A_history['tasks'].copy()

        # Reset current_epoch to last epoch of Task A
        current_epoch = cumulative_history['epoch'][-1]

        # Training data for Task B
        train_data_B = task_splits['B'][task_splits['B']['split'] == 'train']

        # Initialize the model
        model = CustomCNN(num_classes=num_classes_total)
        model.dropout = nn.Dropout(dropout_rate)  # Set dropout rate
        model.to(device)

        # Load model from Task A
        model.load_state_dict(torch.load(model_save_path_A))

        # Define optimizer and criterion
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Sample replay data from Task A
        train_data_A_replay = train_data_A.sample(frac=replay_frac, random_state=42)
        combined_train_data_B = pd.concat([train_data_B, train_data_A_replay]).reset_index(drop=True)

        # Create training dataset and loader for Task B
        combined_train_dataset_B = CustomDataset(combined_train_data_B, transform=transform)
        combined_train_loader_B = DataLoader(combined_train_dataset_B, batch_size=batch_size, shuffle=True)

        # Update model save path for Task B
        model_save_path_B = os.path.join(model_save_dir, f'model_task_B_replay_{int(replay_frac*100)}.pth')

        # Train and validate on Task B with replay
        task_history_B, best_epoch_B = train_and_validate(
            model,
            combined_train_loader_B,
            val_loaders,
            criterion,
            optimizer,
            device,
            num_epochs,
            early_stopping_patience,
            model_save_path_B,
            current_epoch,
            'Task B'
        )

        # Update cumulative history with Task B history
        for key in ['epoch', 'train_loss', 'val_loss_A', 'val_loss_B', 'val_loss_C', 'val_accuracy_A', 'val_accuracy_B', 'val_accuracy_C']:
            cumulative_history[key].extend(task_history_B.get(key, []))
        cumulative_history['tasks'].extend(task_history_B['tasks'])

        # Update current_epoch
        current_epoch = cumulative_history['epoch'][-1]

        # Evaluate on test sets after Task B
        print(f"\nEvaluating on test sets after training on Task B with replay {int(replay_frac*100)}%")
        model.load_state_dict(torch.load(model_save_path_B))
        test_accuracies_B = evaluate_model(model, test_loaders, criterion, device)

        # Store metrics for Task B
        all_metrics[f'Task_B_replay_{int(replay_frac*100)}%'] = {
            'training_history': copy.deepcopy(task_history_B),
            'test_accuracies': test_accuracies_B
        }

        # Release optimizer
        del optimizer
        torch.cuda.empty_cache()

        # Sample replay data from Task A and Task B for Task C
        train_data_B_replay = train_data_B.sample(frac=replay_frac, random_state=42)
        train_data_C = task_splits['C'][task_splits['C']['split'] == 'train']
        combined_train_data_C = pd.concat([train_data_C, train_data_A_replay, train_data_B_replay]).reset_index(drop=True)

        # Create training dataset and loader for Task C
        combined_train_dataset_C = CustomDataset(combined_train_data_C, transform=transform)
        combined_train_loader_C = DataLoader(combined_train_dataset_C, batch_size=batch_size, shuffle=True)

        # Update model save path for Task C
        model_save_path_C = os.path.join(model_save_dir, f'model_task_C_replay_{int(replay_frac*100)}.pth')

        # Define optimizer again for Task C
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # Train and validate on Task C with replay
        task_history_C, best_epoch_C = train_and_validate(
            model,
            combined_train_loader_C,
            val_loaders,
            criterion,
            optimizer,
            device,
            num_epochs,
            early_stopping_patience,
            model_save_path_C,
            current_epoch,
            'Task C'
        )

        # Ensure the final model after Task C training is saved
        final_model_save_path_C = os.path.join(model_save_dir, f'final_model_task_C_replay_{int(replay_frac*100)}.pth')
        torch.save(model.state_dict(), final_model_save_path_C)

        # Update cumulative history with Task C history
        for key in ['epoch', 'train_loss', 'val_loss_A', 'val_loss_B', 'val_loss_C', 'val_accuracy_A', 'val_accuracy_B', 'val_accuracy_C']:
            cumulative_history[key].extend(task_history_C.get(key, []))
        cumulative_history['tasks'].extend(task_history_C['tasks'])

        # Update current_epoch
        current_epoch = cumulative_history['epoch'][-1]

        # Evaluate on test sets after Task C
        print(f"\nEvaluating on test sets after training on Task C with replay {int(replay_frac*100)}%")
        model.load_state_dict(torch.load(model_save_path_C))
        test_accuracies_C = evaluate_model(model, test_loaders, criterion, device)

        # Store metrics for Task C
        all_metrics[f'Task_C_replay_{int(replay_frac*100)}%'] = {
            'training_history': copy.deepcopy(task_history_C),
            'test_accuracies': test_accuracies_C
        }

        # Store cumulative history for this replay percentage
        all_metrics[f'Cumulative_History_{int(replay_frac*100)}%'] = copy.deepcopy(cumulative_history)

        # Release GPU memory
        del model
        del optimizer
        torch.cuda.empty_cache()

    # Save all metrics to a single JSON file
    with open(metrics_file_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
