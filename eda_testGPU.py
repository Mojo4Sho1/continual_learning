import os
import json
import time
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from skimage.io import imread

# Define file paths
train_path = '/home/UNT/jtc0129/Desktop/asl_alphabet_train/asl_alphabet_train'
checkpoint_dir = '/home/UNT/jtc0129/Desktop/checkpoints'
log_dir = '/home/UNT/jtc0129/Desktop/logs'
excluded_classes = ['space', 'nothing', 'del', 'J', 'Z']

# Create checkpoint and log directories if they don't exist
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Function to calculate SSIM using PyTorch, with GPU acceleration
def calculate_ssim_torch(img1, img2, window_size=11, channel=3):
    # Normalize images to [0, 1] range
    img1 = img1 / 255.0
    img2 = img2 / 255.0
    
    # Ensure images are 4D tensors and on the correct device (GPU if available)
    img1 = torch.from_numpy(img1).float().unsqueeze(0).permute(0, 3, 1, 2)  # Convert to PyTorch tensor
    img2 = torch.from_numpy(img2).float().unsqueeze(0).permute(0, 3, 1, 2)  # Convert to PyTorch tensor
    
    # Move images to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img1 = img1.to(device)
    img2 = img2.to(device)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Mean for both images
    window = torch.ones((channel, 1, window_size, window_size)).to(device) / window_size ** 2
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Variance for both images
    sigma1_sq = F.conv2d(img1 ** 2, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # SSIM calculation
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_score = ssim_map.mean().item()

    return ssim_score

# Function to save checkpoint
def save_checkpoint(similar_images, last_processed_image, checkpoint_path):
    checkpoint_data = {
        'similar_images': [list(group) for group in similar_images],  # Convert sets to lists for saving
        'last_processed_image': last_processed_image
    }
    with open(checkpoint_path, 'w') as file:
        json.dump(checkpoint_data, file)
    print(f"Checkpoint saved. Last processed image: {last_processed_image}")

# Function to load checkpoint
def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as file:
            checkpoint_data = json.load(file)
            similar_images = [set(group) for group in checkpoint_data['similar_images']]  # Convert lists back to sets
            last_processed_image = checkpoint_data['last_processed_image']
            return similar_images, last_processed_image
    return [], None  # No checkpoint found

# Function to log results to a JSON file
def save_log(similar_images, log_path):
    with open(log_path, 'w') as file:
        json.dump([list(group) for group in similar_images], file)
    print(f"Log saved to {log_path}")

# Main function to process the classes and compute SSIM
def process_class(class_name):
    class_folder_path = os.path.join(train_path, class_name)
    image_files = os.listdir(class_folder_path)  # Get all images for the class

    checkpoint_path = os.path.join(checkpoint_dir, f"{class_name}_checkpoint.json")
    log_path = os.path.join(log_dir, f"{class_name}_log.json")

    # Load checkpoint if available
    similar_images, last_processed_image = load_checkpoint(checkpoint_path)

    # If resuming, skip already processed images
    if last_processed_image:
        image_files = image_files[image_files.index(last_processed_image) + 1:]

    checkpoint_interval = 100
    similarity_threshold = 0.85  # Set similarity threshold

    # Start processing
    start_time = time.time()
    for count, image_file in enumerate(image_files, start=1):
        image_path = os.path.join(class_folder_path, image_file)
        img = imread(image_path)

        found_set = False

        # Compare to one image from each existing set
        for group in similar_images:
            representative_image = next(iter(group))
            representative_image_path = os.path.join(class_folder_path, representative_image)
            representative_img = imread(representative_image_path)

            # Calculate SSIM using GPU
            similarity_score = calculate_ssim_torch(img, representative_img)

            # If similarity is above threshold, add to this set
            if similarity_score >= similarity_threshold:
                group.add(image_file)
                found_set = True
                break

        # If no match, create a new set
        if not found_set:
            similar_images.append(set([image_file]))

        # Periodically save checkpoints
        if count % checkpoint_interval == 0:
            save_checkpoint(similar_images, image_file, checkpoint_path)

    # Final save after processing the class
    save_log(similar_images, log_path)
    save_checkpoint(similar_images, image_file, checkpoint_path)
    elapsed_time = time.time() - start_time
    print(f"Time taken for class {class_name}: {elapsed_time / 60:.2f} minutes")

# Define tasks with 8 classes each, excluding J and Z
task_A = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
task_B = ['I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q']
task_C = ['R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Notify when each task is completed
def notify_task_completion(task_name):
    print(f"Task {task_name} completed!")

# Process each task
def process_all_tasks():
    for class_name in task_A:
        process_class(class_name)
    notify_task_completion("Task A")

    for class_name in task_B:
        process_class(class_name)
    notify_task_completion("Task B")

    for class_name in task_C:
        process_class(class_name)
    notify_task_completion("Task C")

# Start processing all tasks
process_all_tasks()

