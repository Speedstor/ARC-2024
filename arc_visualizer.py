import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import json
import os

# Create ARC color map
cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)

# Load original data
arc_challenge_file = './input/arc-prize-2025/arc-agi_test_challenges.json'
with open(arc_challenge_file, 'r') as f:
    arc_data = json.load(f)


def visualize_arc_example(train_data, test_data, task_id):
    """Visualize training and test data for an ARC task"""
    # Get number of training and test examples
    n_train = len(train_data)
    n_test = len(test_data)
    
    # Create figure large enough for all examples
    fig, axes = plt.subplots(2, max(n_train, n_test), figsize=(4*max(n_train, n_test), 8))
    fig.suptitle(f"Task ID: {task_id}", fontsize=16)
    
    # Visualize training data
    for i in range(n_train):
        # Input
        axes[0, i].imshow(train_data[i]['input'], cmap=cmap, norm=norm)
        axes[0, i].grid(True, which='both', color='lightgrey', linewidth=0.5)
        axes[0, i].set_title(f"Training #{i+1} - Input")
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        
        # Output
        axes[1, i].imshow(train_data[i]['output'], cmap=cmap, norm=norm)
        axes[1, i].grid(True, which='both', color='lightgrey', linewidth=0.5)
        axes[1, i].set_title(f"Training #{i+1} - Output")
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
    
    # Handle test data visualization
    for i in range(n_test):
        if i < n_train:
            # Already have training data in this column
            pass
        else:
            # Hide unused training cells
            if i >= n_train:
                axes[0, i].axis('off')
                axes[1, i].axis('off')
    
    # Show first test input
    if n_test > 0:
        # Create separate figure for test input
        plt.figure(figsize=(5, 5))
        plt.imshow(test_data[0]['input'], cmap=cmap, norm=norm)
        plt.grid(True, which='both', color='lightgrey', linewidth=0.5)
        plt.title(f"Test Input - {task_id}")
        plt.xticks([])
        plt.yticks([])
        plt.show()

        # Show first test input
        if "output" in test_data[0]:
            # Create separate figure for test input
            plt.figure(figsize=(5, 5))
            plt.imshow(test_data[0]['output'], cmap=cmap, norm=norm)
            plt.grid(True, which='both', color='lightgrey', linewidth=0.5)
            plt.title(f"Test Input - {task_id}")
            plt.xticks([])
            plt.yticks([])
            plt.show()
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()