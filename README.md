# Continual Learning for American Sign Language Recognition

## Project Overview

This project explores continual learning strategies for American Sign Language (ASL) alphabet recognition using deep learning. The primary goal is to address the challenge of **catastrophic forgetting** in neural networks when learning new tasks sequentially.

The project includes:

- **Baseline Model**: Demonstrates forgetting by training sequentially without any mitigation strategies.
- **Replay Models**: Implements replay strategies to mitigate forgetting by rehearsing a fraction of previous task data during new task training.
- **Hyperparameter Tuning**: Uses grid search to find optimal training parameters.
- **Analysis Scripts**: Evaluate model performance and visualize results (not included in the repository).

## Repository Contents

- `eda_testGPU.py`: Script for exploratory data analysis and SSIM comparisions using GPU to improve dataset variability and reduce redundancy between images.
- `gridSearchCNN.py`: Script for hyperparameter tuning using grid search.
- `train_base_forget.py`: Script to train the baseline model demonstrating catastrophic forgetting.
- `train_replay_models.py`: Script to train models using replay strategies.
- `data_splits.csv`: CSV file containing dataset splits for training, validation, and testing.

## Prerequisites

- **Python 3.8+**
- **PyTorch 1.7+**
- **CUDA Toolkit** (for GPU acceleration)
- **Python Libraries**:
  - `torch`
  - `torchvision`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `tqdm`
  - `scikit-learn`
  - `Pillow`


