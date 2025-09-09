#!/usr/bin/env python3

"""
Simple VLMaps Demo Script
This script demonstrates basic VLMaps functionality without complex dependencies.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, '/home/koray/vlmaps')

def check_data_availability():
    """Check if the required data files exist."""
    data_dir = "/home/koray/vlmaps/vlmaps_data_dir/vlmaps_dataset/5LpN3gDmAk7_1"
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Please run the dataset generation script first.")
        return False
    
    map_dir = os.path.join(data_dir, "map_correct")
    if not os.path.exists(map_dir):
        map_dir = os.path.join(data_dir, "map")
    
    if not os.path.exists(map_dir):
        print(f"Map directory not found. Expected: {map_dir}")
        print("Please generate the maps first using the create_map.py script.")
        return False
    
    required_files = [
        "obstacles.npy",
        "color_top_down_1.npy"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(map_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    return True, data_dir, map_dir

def load_map(file_path):
    """Load a numpy map file."""
    try:
        return np.load(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def show_obstacle_map(obstacles_path):
    """Display the obstacle map."""
    print("\n=== Loading Obstacle Map ===")
    obstacles = load_map(obstacles_path)
    
    if obstacles is None:
        return None, None, None, None
    
    # Find the bounds of non-empty areas
    x_indices, y_indices = np.where(obstacles == 0)
    
    if len(x_indices) == 0:
        print("No obstacles found in the map!")
        return None, None, None, None
    
    xmin, xmax = np.min(x_indices), np.max(x_indices)
    ymin, ymax = np.min(y_indices), np.max(y_indices)
    
    print(f"Map bounds: x=[{xmin}, {xmax}], y=[{ymin}, {ymax}]")
    print(f"Unique obstacle values: {np.unique(obstacles)}")
    
    # Crop the map to the relevant area
    obstacles_cropped = obstacles[xmin:xmax+1, ymin:ymax+1]
    
    # Display the obstacle map
    plt.figure(figsize=(10, 8))
    plt.imshow(obstacles_cropped, cmap='gray')
    plt.title("Obstacle Map\n(Black = obstacles, White = free space)")
    plt.colorbar()
    plt.show()
    
    return obstacles, xmin, xmax, ymin, ymax

def show_color_map(color_path, xmin, xmax, ymin, ymax):
    """Display the color top-down map."""
    print("\n=== Loading Color Map ===")
    color_map = load_map(color_path)
    
    if color_map is None:
        return
    
    # Crop to the same bounds as obstacles
    if xmin is not None:
        color_map_cropped = color_map[xmin:xmax+1, ymin:ymax+1]
    else:
        color_map_cropped = color_map
    
    print(f"Color map shape: {color_map_cropped.shape}")
    
    # Display the color map
    plt.figure(figsize=(10, 8))
    plt.imshow(color_map_cropped)
    plt.title("Top-down Color Map")
    plt.axis('off')
    plt.show()

def main():
    """Main function to run the simple demo."""
    print("VLMaps Simple Demo")
    print("==================")
    
    # Check if data is available
    result = check_data_availability()
    if not result or result is False:
        return
    
    success, data_dir, map_dir = result
    if not success:
        return
    
    print(f"Using data directory: {data_dir}")
    print(f"Using map directory: {map_dir}")
    
    # File paths
    obstacles_path = os.path.join(map_dir, "obstacles.npy")
    color_path = os.path.join(map_dir, "color_top_down_1.npy")
    
    # Show obstacle map
    obstacles, xmin, xmax, ymin, ymax = show_obstacle_map(obstacles_path)
    
    if obstacles is not None:
        # Show color map
        show_color_map(color_path, xmin, xmax, ymin, ymax)
        print("\nDemo completed successfully!")
        print("The obstacle map shows navigable areas (white) and obstacles (black).")
        print("The color map shows the top-down view of the environment.")
    else:
        print("Failed to load obstacle map.")

if __name__ == "__main__":
    main()
