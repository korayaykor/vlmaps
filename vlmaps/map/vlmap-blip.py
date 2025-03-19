from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import gdown

from tqdm import tqdm
import cv2
import torchvision.transforms as transforms
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter
import torch

from vlmaps.utils.blip_utils import get_blip_model, get_text_features_multiple_templates, get_lseg_score
from vlmaps.utils.visualize_utils import pool_3d_label_to_2d

from vlmaps.map.vlmap_builder import VLMapBuilder
from vlmaps.map.vlmap_builder_cam import VLMapBuilderCam
from vlmaps.utils.mapping_utils import load_3d_map
from vlmaps.map.map import Map
from vlmaps.utils.index_utils import find_similar_category_id, get_segment_islands_pos, get_dynamic_obstacles_map_3d


class VLMapBLIP(Map):
    def __init__(self, map_config: DictConfig, data_dir: str = ""):
        super().__init__(map_config, data_dir=data_dir)
        self.scores_mat = None
        self.categories = None

    def create_map(self, data_dir: Union[Path, str]) -> None:
        print(f"Creating map for scene at: ", data_dir)
        self._setup_paths(data_dir)
        if self.map_config.pose_info.pose_type == "mobile_base":
            self.map_builder = VLMapBuilder(
                self.data_dir,
                self.map_config,
                self.pose_path,
                self.rgb_paths,
                self.depth_paths,
                self.base2cam_tf,
                self.base_transform,
            )
            self.map_builder.create_mobile_base_map()
        elif self.map_config.pose_info.pose_type == "camera_base":
            self.map_builder = VLMapBuilderCam(
                self.data_dir,
                self.map_config,
                self.pose_path,
                self.rgb_paths,
                self.depth_paths,
                self.base2cam_tf,
                self.base_transform,
            )
            self.map_builder.create_camera_map()
        else:
            raise ValueError("Invalid pose type")

    def load_map(self, data_dir: str) -> bool:
        self._setup_paths(data_dir)
        print(self.data_dir)
        if self.map_config.pose_info.pose_type == "mobile_base":
            self.map_save_path = Path(data_dir) / "vlmap" / "vlmaps.h5df"
            print(self.map_save_path)
            if not self.map_save_path.exists():
                assert False, "Loading VLMap failed because the file doesn't exist."
            (
                self.mapped_iter_list,
                self.grid_feat,
                self.grid_pos,
                self.weight,
                self.occupied_ids,
                self.grid_rgb,
            ) = load_3d_map(self.map_save_path)
        elif self.map_config.pose_info.pose_type == "camera_base":
            self.map_save_path = Path(data_dir) / "vlmap_cam" / "vlmaps_cam.h5df"
            print(self.map_save_path)
            if not self.map_save_path.exists():
                assert False, "Loading VLMap failed because the file doesn't exist."
            (
                self.mapped_iter_list,
                self.grid_feat,
                self.grid_pos,
                self.weight,
                self.occupied_ids,
                self.grid_rgb,
                self.pcd_min,
                self.pcd_max,
                self.cs,
            ) = VLMapBuilderCam.load_3d_map(self.map_save_path)
        else:
            raise ValueError("Invalid pose type")

        return True

    def _init_blip(self):
        if hasattr(self, "blip_model"):
            print("BLIP model is already initialized")
            return
        
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        print("Loading BLIP model...")
        self.blip_model, self.blip_processor, self.blip_feature_model = get_blip_model(self.device)
        
        # Set feature dimension based on BLIP model
        self.blip_feat_dim = self.blip_feature_model.config.hidden_size

    def init_categories(self, categories: List[str]) -> np.ndarray:
        self.categories = categories
        self._init_blip()  # Make sure BLIP is initialized
        
        # Get scores using BLIP for each category
        self.scores_mat = get_lseg_score(
            self.blip_model,
            self.blip_processor,
            self.blip_feature_model,
            self.categories,
            self.grid_feat,
            use_multiple_templates=True,
            add_other=True,
            device=self.device
        )
        
        return self.scores_mat

    def index_map(self, language_desc: str, with_init_cat: bool = True):
        if with_init_cat and self.scores_mat is not None and self.categories is not None:
            cat_id = find_similar_category_id(language_desc, self.categories)
            scores_mat = self.scores_mat
        else:
            if with_init_cat:
                raise Exception(
                    "Categories are not preloaded. Call init_categories(categories: List[str]) to initialize categories."
                )
            
            # Initialize BLIP if not already done
            if not hasattr(self, "blip_model"):
                self._init_blip()
                
            # Get scores for this specific language description
            scores_mat = get_lseg_score(
                self.blip_model,
                self.blip_processor,
                self.blip_feature_model,
                [language_desc],
                self.grid_feat,
                use_multiple_templates=True,
                add_other=True,
                device=self.device
            )
            cat_id = 0

        max_ids = np.argmax(scores_mat, axis=1)
        mask = max_ids == cat_id
        return mask

    def customize_obstacle_map(
        self,
        potential_obstacle_names: List[str],
        obstacle_names: List[str],
        vis: bool = False,
    ):
        if self.obstacles_cropped is None and self.obstacles_map is None:
            self.generate_obstacle_map()
            
        # Initialize BLIP if not already done
        if not hasattr(self, "blip_model"):
            self._init_blip()

        # Adapt get_dynamic_obstacles_map_3d to use BLIP instead of CLIP
        self.obstacles_new_cropped = get_dynamic_obstacles_map_3d_blip(
            self.blip_model,
            self.blip_processor,
            self.blip_feature_model,
            self.obstacles_cropped,
            self.map_config.potential_obstacle_names,
            self.map_config.obstacle_names,
            self.grid_feat,
            self.grid_pos,
            self.rmin,
            self.cmin,
            self.blip_feat_dim,
            device=self.device,
            vis=vis,
        )
        
        self.obstacles_new_cropped = Map._dilate_map(
            self.obstacles_new_cropped == 0,
            self.map_config.dilate_iter,
            self.map_config.gaussian_sigma,
        )
        self.obstacles_new_cropped = self.obstacles_new_cropped == 0

    def get_pos(self, name: str) -> Tuple[List[List[int]], List[List[float]], List[np.ndarray], Any]:
        """
        Get the contours, centers, and bbox list of a certain category
        on a full map
        """
        assert self.categories
        pc_mask = self.index_map(name, with_init_cat=True)
        mask_2d = pool_3d_label_to_2d(pc_mask, self.grid_pos, self.gs)
        mask_2d = mask_2d[self.rmin : self.rmax + 1, self.cmin : self.cmax + 1]

        foreground = binary_closing(mask_2d, iterations=3)
        foreground = gaussian_filter(foreground.astype(float), sigma=0.8, truncate=3)
        foreground = foreground > 0.5
        foreground = binary_dilation(foreground)

        contours, centers, bbox_list, _ = get_segment_islands_pos(foreground, 1)

        # whole map position
        for i in range(len(contours)):
            centers[i][0] += self.rmin
            centers[i][1] += self.cmin
            bbox_list[i][0] += self.rmin
            bbox_list[i][1] += self.rmin
            bbox_list[i][2] += self.cmin
            bbox_list[i][3] += self.cmin
            for j in range(len(contours[i])):
                contours[i][j, 0] += self.rmin
                contours[i][j, 1] += self.cmin

        return contours, centers, bbox_list


def get_dynamic_obstacles_map_3d_blip(
    blip_model,
    blip_processor,
    blip_feature_model,
    obstacles_cropped,
    potential_obstacle_classes,
    obstacle_classes,
    grid_feat,
    grid_pos,
    rmin,
    cmin,
    blip_feat_dim,
    device="cuda",
    vis=False,
):
    """
    BLIP version of get_dynamic_obstacles_map_3d
    """
    all_obstacles_mask = obstacles_cropped == 0
    
    # Get scores using BLIP
    scores_mat = get_lseg_score(
        blip_model,
        blip_processor,
        blip_feature_model,
        potential_obstacle_classes,
        grid_feat,
        use_multiple_templates=True,
        add_other=False,
        device=device
    )
    
    predict = np.argmax(scores_mat, axis=1)
    obs_inds = []
    for obs_name in obstacle_classes:
        for i, po_obs_name in enumerate(potential_obstacle_classes):
            if obs_name == po_obs_name:
                obs_inds.append(i)

    print("obs_inds: ", obs_inds)
    pts_mask = np.zeros_like(predict, dtype=bool)
    for id in obs_inds:
        tmp = predict == id
        pts_mask = np.logical_or(pts_mask, tmp)

    # new_obstacles = obstacles_segment_map != 1
    new_obstacles = np.zeros_like(obstacles_cropped, dtype=bool)
    obs_pts = grid_pos[pts_mask]
    mask1 = np.logical_and(obs_pts[:, 0] - rmin >= 0, obs_pts[:, 1] - cmin >= 0) 
    mask2 = np.logical_and(obs_pts[:, 0] - rmin < new_obstacles.shape[0], obs_pts[ :, 1] - cmin < new_obstacles.shape[1])
    mask = np.logical_and(mask1, mask2)
    new_obstacles[obs_pts[mask, 0] - rmin, obs_pts[mask, 1] - cmin] = 1
    new_obstacles[obs_pts[:, 0] - rmin, obs_pts[:, 1] - cmin] = 1
    new_obstacles = np.logical_and(new_obstacles, all_obstacles_mask)
    new_obstacles = np.logical_not(new_obstacles)

    if vis:
        cv2.imshow("new obstacles_cropped", (new_obstacles * 255).astype(np.uint8))
        cv2.waitKey()
    return new_obstacles
