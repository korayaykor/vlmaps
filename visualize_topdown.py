#!/usr/bin/env python3
"""
VLMaps Top-Down Visualization Script
This script loads a VLMaps h5df file and creates a top-down RGB visualization.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

def load_vlmap(vlmap_path):
    """Load VLMap from h5df file."""
    print(f"Loading VLMap from: {vlmap_path}")
    
    with h5py.File(vlmap_path, 'r') as f:
        print("\nAvailable keys in h5df file:")
        for key in f.keys():
            print(f"  - {key}: shape={f[key].shape}, dtype={f[key].dtype}")
        
        grid_feat = np.array(f['grid_feat'])
        grid_pos = np.array(f['grid_pos'])
        grid_rgb = np.array(f['grid_rgb'])
        weight = np.array(f['weight'])
        
    return grid_feat, grid_pos, grid_rgb, weight

def create_topdown_map(grid_pos, grid_rgb, grid_size=480):
    """Create a top-down RGB map from 3D grid data."""
    print(f"\nCreating top-down map with grid size: {grid_size}x{grid_size}")
    
    # Initialize the 2D maps
    topdown_rgb = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    height_map = np.full((grid_size, grid_size), -np.inf, dtype=np.float32)
    
    # Fill the maps
    for i, (pos, rgb) in enumerate(zip(grid_pos, grid_rgb)):
        row, col, height = pos
        
        # Skip invalid positions
        if row < 0 or row >= grid_size or col < 0 or col >= grid_size:
            continue
        
        # Keep the highest point at each (row, col)
        if height > height_map[row, col]:
            height_map[row, col] = height
            topdown_rgb[row, col] = rgb
    
    return topdown_rgb, height_map

def create_obstacle_map(grid_pos, height_map, grid_size=480, obstacle_height_range=(0, 50)):
    """Create a simple obstacle map based on height and occupancy."""
    obstacle_map = np.ones((grid_size, grid_size), dtype=np.uint8) * 255  # Start with all free (white)
    
    # Mark occupied cells as obstacles (black)
    min_height, max_height = obstacle_height_range
    for pos in grid_pos:
        row, col, height = pos
        if 0 <= row < grid_size and 0 <= col < grid_size:
            if min_height <= height <= max_height:
                obstacle_map[row, col] = 0  # Mark as obstacle
    
    return obstacle_map

def visualize_maps(topdown_rgb, obstacle_map, height_map, save_dir=None):
    """Visualize the top-down maps."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Top-down RGB map
    axes[0].imshow(topdown_rgb)
    axes[0].set_title("Top-Down RGB Map")
    axes[0].axis('off')
    
    # Obstacle map
    axes[1].imshow(obstacle_map, cmap='gray')
    axes[1].set_title("Obstacle Map\n(Black=obstacle, White=free)")
    axes[1].axis('off')
    
    # Height map
    height_map_vis = height_map.copy()
    height_map_vis[height_map_vis == -np.inf] = np.nan
    im = axes[2].imshow(height_map_vis, cmap='viridis')
    axes[2].set_title("Height Map")
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Save individual maps
        plt.savefig(save_dir / "all_maps.png", dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to: {save_dir / 'all_maps.png'}")
        
        # Save individual arrays
        np.save(save_dir / "topdown_rgb.npy", topdown_rgb)
        np.save(save_dir / "obstacle_map.npy", obstacle_map)
        np.save(save_dir / "height_map.npy", height_map)
        print(f"Saved numpy arrays to: {save_dir}")
    
    plt.show()

def main():
    # Default path
    vlmap_path = "/home/koray/vlmaps/vlmaps_data_dir/vlmaps_dataset/5LpN3gDmAk7_1/vlmap/vlmaps.h5df"
    
    # Check if path is provided as argument
    if len(sys.argv) > 1:
        vlmap_path = sys.argv[1]
    
    if not Path(vlmap_path).exists():
        print(f"Error: VLMap file not found at: {vlmap_path}")
        print("\nUsage: python visualize_topdown.py [path_to_vlmaps.h5df]")
        return
    
    # Load the VLMap
    grid_feat, grid_pos, grid_rgb, weight = load_vlmap(vlmap_path)
    
    print(f"\nVLMap statistics:")
    print(f"  Number of occupied voxels: {len(grid_pos)}")
    print(f"  Feature dimension: {grid_feat.shape[1]}")
    print(f"  Position bounds:")
    print(f"    Row: [{grid_pos[:, 0].min()}, {grid_pos[:, 0].max()}]")
    print(f"    Col: [{grid_pos[:, 1].min()}, {grid_pos[:, 1].max()}]")
    print(f"    Height: [{grid_pos[:, 2].min()}, {grid_pos[:, 2].max()}]")
    
    # Determine grid size from the data
    grid_size = max(grid_pos[:, 0].max(), grid_pos[:, 1].max()) + 1
    print(f"  Inferred grid size: {grid_size}")
    
    # Create top-down maps
    topdown_rgb, height_map = create_topdown_map(grid_pos, grid_rgb, grid_size=grid_size)
    obstacle_map = create_obstacle_map(grid_pos, height_map, grid_size=grid_size)
    
    # Determine save directory
    vlmap_dir = Path(vlmap_path).parent
    save_dir = vlmap_dir / "visualizations"
    
    # Visualize and save
    visualize_maps(topdown_rgb, obstacle_map, height_map, save_dir=save_dir)
    
    print("\n✓ Visualization complete!")
    print(f"\nTo use these maps in your navigation code:")
    print(f"  topdown_rgb = np.load('{save_dir}/topdown_rgb.npy')")
    print(f"  obstacle_map = np.load('{save_dir}/obstacle_map.npy')")
    print(f"  height_map = np.load('{save_dir}/height_map.npy')")

if __name__ == "__main__":
    main()
