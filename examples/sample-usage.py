from pathlib import Path
import hydra
from omegaconf import DictConfig
import cv2
import numpy as np

from vlmaps.map.vlmap_blip import VLMapBLIP
from vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.utils.visualize_utils import (
    pool_3d_label_to_2d,
    pool_3d_rgb_to_2d,
    visualize_rgb_map_3d,
    visualize_masked_map_2d,
    visualize_heatmap_2d,
    visualize_masked_map_3d,
    visualize_heatmap_3d,
    get_heatmap_from_mask_2d,
    get_heatmap_from_mask_3d,
)


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="map_indexing_cfg.yaml",
)
def main(config: DictConfig) -> None:
    data_dir = Path(config.data_paths.vlmaps_data_dir)
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])
    print(data_dirs[config.scene_id])
    
    # Create VLMapBLIP instead of VLMap
    vlmap = VLMapBLIP(config.map_config, data_dir=data_dirs[config.scene_id])
    
    # Load map data
    vlmap.load_map(data_dirs[config.scene_id])
    
    # Visualize the 3D RGB map
    visualize_rgb_map_3d(vlmap.grid_pos, vlmap.grid_rgb)
    
    # Ask for a category to search
    cat = input("What is your interested category in this scene?")
    
    # Initialize BLIP model
    vlmap._init_blip()
    
    print("considering categories: ")
    print(mp3dcat[1:-1])
    
    # Initialize categories (needed for index_map)
    if config.init_categories:
        vlmap.init_categories(mp3dcat[1:-1])
        mask = vlmap.index_map(cat, with_init_cat=True)
    else:
        mask = vlmap.index_map(cat, with_init_cat=False)

    # Visualize results in 2D or 3D based on config
    if config.index_2d:
        mask_2d = pool_3d_label_to_2d(mask, vlmap.grid_pos, config.params.gs)
        rgb_2d = pool_3d_rgb_to_2d(vlmap.grid_rgb, vlmap.grid_pos, config.params.gs)
        visualize_masked_map_2d(rgb_2d, mask_2d)
        heatmap = get_heatmap_from_mask_2d(mask_2d, cell_size=config.params.cs, decay_rate=config.decay_rate)
        visualize_heatmap_2d(rgb_2d, heatmap)
    else:
        visualize_masked_map_3d(vlmap.grid_pos, mask, vlmap.grid_rgb)
        heatmap = get_heatmap_from_mask_3d(
            vlmap.grid_pos, mask, cell_size=config.params.cs, decay_rate=config.decay_rate
        )
        visualize_heatmap_3d(vlmap.grid_pos, heatmap, vlmap.grid_rgb)


if __name__ == "__main__":
    main()
