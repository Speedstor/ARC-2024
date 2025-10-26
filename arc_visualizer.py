import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.lines as mlines # Import lines for the new method
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
    """
    Visualize training and test data for an ARC task.
    
    Plots training examples side-by-side, followed by the first
    test example in the last column.
    A vertical red line is drawn in the whitespace BETWEEN them.
    """
    n_train = len(train_data)
    n_test = len(test_data)
    has_test = n_test > 0
    
    n_cols = n_train + (1 if has_test else 0)
    
    if n_cols == 0:
        print(f"No training or test data for task {task_id}")
        return

    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))
    fig.suptitle(f"Task ID: {task_id}", fontsize=16)
    
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    # --- 4. Visualize training data ---
    for i in range(n_train):
        axes[0, i].imshow(train_data[i]['input'], cmap=cmap, norm=norm)
        axes[0, i].grid(True, which='both', color='lightgrey', linewidth=0.5)
        axes[0, i].set_title(f"Training #{i+1} - Input")
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        
        axes[1, i].imshow(train_data[i]['output'], cmap=cmap, norm=norm)
        axes[1, i].grid(True, which='both', color='lightgrey', linewidth=0.5)
        axes[1, i].set_title(f"Training #{i+1} - Output")
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

    # --- 5. Visualize test data (if available) ---
    if has_test:
        test_col = n_train 
        has_output = "output" in test_data[0]
        
        # Plot Test Input
        axes[0, test_col].imshow(test_data[0]['input'], cmap=cmap, norm=norm)
        axes[0, test_col].grid(True, which='both', color='lightgrey', linewidth=0.5)
        axes[0, test_col].set_title(f"Test #1 - Input")
        axes[0, test_col].set_xticks([])
        axes[0, test_col].set_yticks([])
        
        # Plot Test Output
        if has_output:
            axes[1, test_col].imshow(test_data[0]['output'], cmap=cmap, norm=norm)
            axes[1, test_col].grid(True, which='both', color='lightgrey', linewidth=0.5)
            axes[1, test_col].set_title(f"Test #1 - Output")
            axes[1, test_col].set_xticks([])
            axes[1, test_col].set_yticks([])
        else:
            axes[1, test_col].axis('off')
            axes[1, test_col].set_title("Test #1 - Output (N/A)")

    # --- 7. Show the combined figure ---
    # We must call tight_layout() BEFORE getting positions,
    # as it adjusts the subplot sizes and locations.
    plt.tight_layout()
    plt.subplots_adjust(top=0.9) 
    
    # --- 6. Add the vertical separator line (NEW METHOD) ---
    # This now happens *after* tight_layout
    if has_test and n_train > 0:
        # Get the axes for the last training plot and the test plot
        ax_train_last = axes[0, n_train - 1]
        ax_test_first = axes[0, test_col]
        
        # Get position of the axes in "figure coordinates" (0 to 1)
        # .get_position() returns a Bbox [x0, y0, width, height]
        pos_train = ax_train_last.get_position()
        pos_test = ax_test_first.get_position()
        
        # Get the y-coordinates from the top/bottom axes
        # We'll use the bottom of the bottom-row plot
        # and the top of the top-row plot
        y_bottom = axes[1, 0].get_position().y0
        y_top = axes[0, 0].get_position().y1
        
        # Calculate the x-coordinate for the line:
        # halfway between the right edge of the train plot (pos_train.x1)
        # and the left edge of the test plot (pos_test.x0)
        line_x = (pos_train.x1 + pos_test.x0) / 2
        
        # Create the line in figure coordinates
        line = mlines.Line2D(
            [line_x, line_x],        # x_start, x_end
            [y_bottom, y_top],     # y_start, y_end
            transform=fig.transFigure, # Use figure coordinates
            color='black',
            linewidth=3
        )
        
        # Add the line artist to the figure
        fig.add_artist(line)

    plt.show()