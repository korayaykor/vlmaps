#!/usr/bin/env python3
"""
Generate Top-Down Semantic Maps from VLMaps
This script creates semantic heatmaps for specific object categories.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from tqdm import tqdm

# Add vlmaps to path
sys.path.insert(0, '/home/koray/vlmaps')

from vlmaps.map.vlmap import VLMap
from omegaconf import OmegaConf

def pool_3d_label_to_2d(mask_3d, grid_pos, grid_size):
    """Pool 3D semantic labels to 2D top-down view."""
    mask_2d = np.zeros((grid_size, grid_size), dtype=np.float32)
    count_2d = np.zeros((grid_size, grid_size), dtype=np.int32)
    
    for i, (pos, mask_val) in enumerate(zip(grid_pos, mask_3d)):
        row, col, height = pos
        if 0 <= row < grid_size and 0 <= col < grid_size:
            mask_2d[row, col] += mask_val
            count_2d[row, col] += 1
    
    # Average the values
    mask_2d = np.divide(mask_2d, count_2d, where=count_2d > 0)
    return mask_2d

def pool_3d_rgb_to_2d(grid_rgb, grid_pos, grid_size):
    """Pool 3D RGB values to 2D top-down view."""
    rgb_2d = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    height_map = np.full((grid_size, grid_size), -np.inf)
    
    for pos, rgb in zip(grid_pos, grid_rgb):
        row, col, height = pos
        if 0 <= row < grid_size and 0 <= col < grid_size:
            if height > height_map[row, col]:
                height_map[row, col] = height
                rgb_2d[row, col] = rgb
    
    return rgb_2d

def create_heatmap_from_mask(mask_2d, cell_size=0.05, decay_rate=5.0):
    """Create a heatmap with distance decay from semantic mask."""
    from scipy.ndimage import distance_transform_edt
    
    # Create distance transform
    binary_mask = (mask_2d > 0.5).astype(np.uint8)
    distances = distance_transform_edt(1 - binary_mask) * cell_size
    
    # Apply exponential decay
    heatmap = np.exp(-decay_rate * distances)
    heatmap = heatmap * mask_2d.max()  # Scale by max confidence
    
    return heatmap

def generate_semantic_maps(vlmap_path, categories, grid_size=480, cell_size=0.05):
    """Generate semantic maps for multiple categories."""
    
    print(f"Loading VLMap from: {vlmap_path}")
    
    # Create a minimal config
    config = OmegaConf.create({
        'camera_height': 1.5,
        'cs': cell_size,
        'gs': grid_size,
        'map_boundary_factor': 1.0,
        'data_dir': str(Path(vlmap_path).parent.parent)
    })
    
    # Load the map
    data_dir = Path(vlmap_path).parent.parent
    vlmap = VLMap(config, data_dir=data_dir)
    
    # Load map data directly
    with h5py.File(vlmap_path, 'r') as f:
        grid_feat = np.array(f['grid_feat'])
        grid_pos = np.array(f['grid_pos'])
        grid_rgb = np.array(f['grid_rgb'])
    
    vlmap.grid_feat = grid_feat
    vlmap.grid_pos = grid_pos
    vlmap.grid_rgb = grid_rgb
    
    print(f"Map loaded: {len(grid_pos)} voxels")
    
    # Initialize CLIP for semantic queries
    print("Initializing CLIP model...")
    vlmap._init_clip()
    
    # Create RGB top-down map
    actual_grid_size = max(grid_pos[:, 0].max(), grid_pos[:, 1].max()) + 1
    rgb_2d = pool_3d_rgb_to_2d(grid_rgb, grid_pos, actual_grid_size)
    
    # Generate semantic maps for each category
    results = {}
    save_dir = Path(vlmap_path).parent / "semantic_maps"
    save_dir.mkdir(exist_ok=True, parents=True)
    
    for category in tqdm(categories, desc="Generating semantic maps"):
        print(f"\nProcessing category: {category}")
        
        # Query the map for this category
        mask_3d = vlmap.index_map(category, with_init_cat=False)
        
        # Pool to 2D
        mask_2d = pool_3d_label_to_2d(mask_3d, grid_pos, actual_grid_size)
        
        # Create heatmap
        heatmap = create_heatmap_from_mask(mask_2d, cell_size=cell_size)
        
        # Store results
        results[category] = {
            'mask_2d': mask_2d,
            'heatmap': heatmap
        }
        
        # Save numpy arrays
        np.save(save_dir / f"{category}_mask.npy", mask_2d)
        np.save(save_dir / f"{category}_heatmap.npy", heatmap)
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # RGB map
        axes[0].imshow(rgb_2d)
        axes[0].set_title(f"RGB Top-Down Map")
        axes[0].axis('off')
        
        # Semantic mask
        axes[1].imshow(rgb_2d)
        axes[1].imshow(mask_2d, alpha=0.6, cmap='hot')
        axes[1].set_title(f"Semantic Mask: {category}")
        axes[1].axis('off')
        
        # Heatmap
        axes[2].imshow(rgb_2d)
        axes[2].imshow(heatmap, alpha=0.6, cmap='hot')
        axes[2].set_title(f"Heatmap: {category}")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / f"{category}_visualization.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_dir / f'{category}_visualization.png'}")
    
    # Save RGB map
    np.save(save_dir / "rgb_topdown.npy", rgb_2d)
    
    print(f"\n✓ Semantic maps generated successfully!")
    print(f"  Saved to: {save_dir}")
    print(f"\nGenerated maps for: {', '.join(categories)}")
    
    return results, rgb_2d, save_dir

def main():
    # Configuration
    vlmap_path = "/home/koray/vlmaps/vlmaps_data_dir/vlmaps_dataset/5LpN3gDmAk7_1/vlmap/vlmaps.h5df"
    
    # Categories to query (you can customize this list)
    categories = [
        "chair",
        "table", 
        "sofa",
        "bed",
        "door",
        "window",
        "plant"
    ]
    
    # Allow command-line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("Usage: python generate_semantic_map.py [category1] [category2] ...")
            print("\nExample: python generate_semantic_map.py chair table sofa")
            print("\nDefault categories:", categories)
            return
        categories = sys.argv[1:]
    
    if not Path(vlmap_path).exists():
        print(f"Error: VLMap not found at {vlmap_path}")
        print("Please run create_map.py first to generate the VLMap.")
        return
    
    # Generate semantic maps
    results, rgb_2d, save_dir = generate_semantic_maps(vlmap_path, categories)
    
    print("\n" + "="*60)
    print("How to use the generated semantic maps:")
    print("="*60)
    print(f"\n# Load RGB top-down map:")
    print(f"rgb_map = np.load('{save_dir}/rgb_topdown.npy')")
    print(f"\n# Load semantic mask for a category (e.g., 'chair'):")
    print(f"chair_mask = np.load('{save_dir}/chair_mask.npy')")
    print(f"\n# Load heatmap (with distance decay):")
    print(f"chair_heatmap = np.load('{save_dir}/chair_heatmap.npy')")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
