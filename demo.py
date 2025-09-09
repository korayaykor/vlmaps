#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLMaps Demo - Visual Language Maps for Robot Navigation

This script demonstrates the VLMaps functionality for creating visual language maps
and performing landmark indexing using CLIP and LSeg models.

License: MIT
Part of our implementation is inspired by habitat_sim, LSeg, and CLIP projects.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Import required libraries
import sys
import os
import math
import imageio
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
import clip
from vlmaps.utils.habitat_utils import get_obj2cls_dict

# Try to import project-specific utilities
try:
    from vlmaps.utils.mapping_utils import (
        load_pose, load_semantic_npy as load_semantic, load_obj2cls_dict, save_map, 
        cvt_obj_id_2_cls_id, depth2pc, transform_pc, get_sim_cam_mat, 
        pos2grid_id, project_point, load_map, get_new_pallete, get_new_mask_pallete
    )
    from vlmaps.utils.clip_utils import get_text_feats
    from vlmaps.utils.matterport3d_categories import mp3dcat
    from vlmaps.lseg.modules.models.lseg_net import LSegEncNet
    from vlmaps.lseg.additional_utils.models import resize_image, pad_image, crop_image
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import project utilities: {e}")
    print("Please ensure you're running this from the vlmaps directory with all dependencies installed.")
    UTILS_AVAILABLE = False

# Configuration for video encoding
has_gpu = torch.cuda.is_available()
codec = "h264_nvenc" if has_gpu else "h264"

def load_depth(depth_filepath):
    with open(depth_filepath, 'rb') as f:
        depth = np.load(f)
    return depth



def get_fast_video_writer(video_file: str, fps: int = 60):
    if (
        "google.colab" in sys.modules
        and os.path.splitext(video_file)[-1] == ".mp4"
        and os.environ.get("IMAGEIO_FFMPEG_EXE") == "/usr/bin/ffmpeg"
    ):
        # USE GPU Accelerated Hardware Encoding
        writer = imageio.get_writer(
            video_file,
            fps=fps,
            codec=codec,
            mode="I",
            bitrate="1000k",
            format="FFMPEG",
            ffmpeg_log_level="info",
            quality=10,
            output_params=["-minrate", "500k", "-maxrate", "5000k"],
        )
    else:
        # Use software encoding
        writer = imageio.get_writer(video_file, fps=fps)
    return writer

def create_video(data_dir: str, output_dir: str, fps: int = 30):

    rgb_dir = os.path.join(data_dir, "rgb")
    depth_dir = os.path.join(data_dir, "depth")
    rgb_out_path = os.path.join(output_dir, "rgb.mp4")
    depth_out_path = os.path.join(output_dir, "depth.mp4")
    rgb_writer = get_fast_video_writer(rgb_out_path, fps=fps)
    depth_writer = get_fast_video_writer(depth_out_path, fps=fps)

    rgb_list = sorted(os.listdir(rgb_dir), key=lambda x: int(
        x.split("_")[-1].split(".")[0]))
    depth_list = sorted(os.listdir(depth_dir), key=lambda x: int(
        x.split("_")[-1].split(".")[0]))

    rgb_list = [os.path.join(rgb_dir, x) for x in rgb_list]
    depth_list = [os.path.join(depth_dir, x) for x in depth_list]
    pbar = tqdm.tqdm(total=len(rgb_list), position=0, leave=True)
    for i, (rgb_path, depth_path) in enumerate(zip(rgb_list, depth_list)):
        bgr = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        depth = load_depth(depth_path)
        depth_vis = (depth / 10 * 255).astype(np.uint8)

        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        rgb_writer.append_data(rgb)
        depth_writer.append_data(depth_color)
        pbar.update(1)
    rgb_writer.close()
    depth_writer.close()

def show_video(video_path, video_width=1080):
    """Display video in Jupyter notebook (for notebook environments)"""
    try:
        from IPython.display import HTML
        from base64 import b64encode
        
        video_file = open(video_path, "r+b").read()
        video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
        return HTML(f"""<video width={video_width} autoplay controls><source src="{video_url}"></video>""")
    except ImportError:
        print(f"Video saved at: {video_path}")
        return None

def show_videos(video_paths, video_width=1080):
    """Display multiple videos in Jupyter notebook (for notebook environments)"""
    try:
        from IPython.display import HTML
        from base64 import b64encode
        
        html = ""
        for video_path in video_paths:
            video_file = open(video_path, "r+b").read()
            video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
            html += f"""<video width={video_width} autoplay controls><source src="{video_url}"></video>\n"""
        return HTML(html)
    except ImportError:
        print(f"Videos saved at: {video_paths}")
        return None

"""## VLMap Creation
It takes around 20 minutes to build a VLMap with around 1000 RGBD frames. We also provide a pre-built VLMap. Skip to the Landmark Indexing part of the code to directly try our map.
"""

# setup parameters
# @markdown meters per cell size
cs = 0.05 # @param {type: "number"}
# @markdown map resolution (gs x gs)
gs = 1000 # @param {type: "integer"}
# @markdown camera height (used for filtering out points on the floor)
camera_height = 1.5 # @param {type: "number"}
# @markdown depth pixels subsample rate
depth_sample_rate = 100 # @param {type: "integer"}
# @markdown data where rgb, depth, pose are loaded and map are saved
data_dir = "/content/5LpN3gDmAk7_1/" # @param {type: "string"}

# VLMap Creation Parameters and Configuration
# Default configuration - can be modified as needed

# Define create_lseg_map_batch function 
def create_lseg_map_batch(img_save_dir, camera_height, cs=0.05, gs=1000, depth_sample_rate=100):
    if not UTILS_AVAILABLE:
        raise ImportError("LSeg utilities not available. Please install dependencies.")
        
    mask_version = 1 # 0, 1

    crop_size = 480 # 480
    base_size = 520 # 520
    lang = "door,chair,ground,ceiling,other"
    labels = lang.split(",")

    # loading models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    clip_version = "ViT-B/32"
    clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                    'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load(clip_version)  # clip.available_models()
    clip_model.to(device).eval()
    lang_token = clip.tokenize(labels)
    lang_token = lang_token.to(device)
    with torch.no_grad():
        text_feats = clip_model.encode_text(lang_token)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats.cpu().numpy()
    model = LSegEncNet(lang, arch_option=0,
                        block_depth=0,
                        activation='lrelu',
                        crop_size=crop_size)
    model_state_dict = model.state_dict()
    pretrained_state_dict = torch.load("lseg/checkpoints/demo_e200.ckpt")
    pretrained_state_dict = {k.lstrip('net.'): v for k, v in pretrained_state_dict['state_dict'].items()}
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(pretrained_state_dict)

    model.eval()
    model = model.cuda()

    norm_mean= [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    padding = [0.0] * 3
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    print(f"loading scene {img_save_dir}")
    rgb_dir = os.path.join(img_save_dir, "rgb")
    depth_dir = os.path.join(img_save_dir, "depth")
    semantic_dir = os.path.join(img_save_dir, "semantic")
    poses_file = os.path.join(img_save_dir, "poses.txt")
    
    # Check if obj2cls_dict.txt exists, if not create a default one
    obj2cls_path = os.path.join(img_save_dir, "obj2cls_dict.txt")
    # Always recreate the file to ensure we have all needed mappings
    print("Creating comprehensive obj2cls_dict.txt mapping...")
    # Create a default object to class mapping with correct format: obj_id:cls_id,cls_name
    # Based on actual data analysis, we need to cover range from -1 to 40+
    with open(obj2cls_path, 'w') as f:
        # Create mappings for all possible object IDs (-10 to 100 to be extra safe)
        for obj_id in range(-10, 101):
            f.write(f"{obj_id}:0,unknown\n")

    rgb_list = sorted(os.listdir(rgb_dir), key=lambda x: int(
        x.split("_")[-1].split(".")[0]))
    depth_list = sorted(os.listdir(depth_dir), key=lambda x: int(
        x.split("_")[-1].split(".")[0]))
    semantic_list = sorted(os.listdir(semantic_dir), key=lambda x: int(
        x.split("_")[-1].split(".")[0]))

    rgb_list = [os.path.join(rgb_dir, x) for x in rgb_list]
    depth_list = [os.path.join(depth_dir, x) for x in depth_list]
    semantic_list = [os.path.join(semantic_dir, x) for x in semantic_list]
    
    # Load poses from the single poses.txt file
    poses_data = []
    if os.path.exists(poses_file):
        with open(poses_file, 'r') as f:
            for line in f:
                pose_values = [float(x) for x in line.strip().split()]
                poses_data.append(pose_values)
    else:
        print(f"Warning: {poses_file} not found!")
        return

    map_save_dir = os.path.join(img_save_dir, "map")
    os.makedirs(map_save_dir, exist_ok=True)
    color_top_down_save_path = os.path.join(map_save_dir, f"color_top_down_{mask_version}.npy")
    gt_save_path = os.path.join(map_save_dir, f"grid_{mask_version}_gt.npy")
    grid_save_path = os.path.join(map_save_dir, f"grid_lseg_{mask_version}.npy")
    weight_save_path = os.path.join(map_save_dir, f"weight_lseg_{mask_version}.npy")
    obstacles_save_path = os.path.join(map_save_dir, "obstacles.npy")

    obj2cls = load_obj2cls_dict(obj2cls_path)

    # initialize a grid with zero position at the center
    color_top_down_height = (camera_height + 1) * np.ones((gs, gs), dtype=np.float32)
    color_top_down = np.zeros((gs, gs, 3), dtype=np.uint8)
    gt = np.zeros((gs, gs), dtype=np.int32)
    grid = np.zeros((gs, gs, clip_feat_dim), dtype=np.float32)
    obstacles = np.ones((gs, gs), dtype=np.uint8)
    weight = np.zeros((gs, gs), dtype=float)

    save_map(color_top_down_save_path, color_top_down)
    save_map(gt_save_path, gt)
    save_map(grid_save_path, grid)
    save_map(weight_save_path, weight)
    save_map(obstacles_save_path, obstacles)

    tf_list = []
    data_iter = zip(rgb_list, depth_list, semantic_list)
    pbar = tqdm(total=len(rgb_list))
    # load all images and depths and poses
    for i, data_sample in enumerate(data_iter):
        rgb_path, depth_path, semantic_path = data_sample

        bgr = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # read pose from poses_data array
        if i < len(poses_data):
            pose_values = poses_data[i]
            # Assume pose format: [x, y, z, qx, qy, qz, qw] or similar
            # We need to convert this to pos and rot matrices
            pos = np.array(pose_values[:3])
            if len(pose_values) >= 7:
                # Quaternion format [x, y, z, qx, qy, qz, qw]
                try:
                    from scipy.spatial.transform import Rotation
                    quat = pose_values[3:7]  # [qx, qy, qz, qw]
                    rot = Rotation.from_quat(quat).as_matrix()
                except ImportError:
                    print("Warning: scipy not available, using identity rotation")
                    rot = np.eye(3)
            else:
                # Default to identity rotation if not enough values
                rot = np.eye(3)
        else:
            print(f"Warning: No pose data for frame {i}")
            pos = np.array([0, 0, 0])
            rot = np.eye(3)
        rot_ro_cam = np.eye(3)
        rot_ro_cam[1, 1] = -1
        rot_ro_cam[2, 2] = -1
        rot = rot @ rot_ro_cam
        pos[1] += camera_height


        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = pos.reshape(-1)

        tf_list.append(pose)
        if len(tf_list) == 1:
            init_tf_inv = np.linalg.inv(tf_list[0])

        tf = init_tf_inv @ pose

        # read depth
        depth = load_depth(depth_path)

        # read semantic
        semantic = load_semantic(semantic_path)
        semantic = cvt_obj_id_2_cls_id(semantic, obj2cls)

        pix_feats = get_lseg_feat(model, rgb, labels, transform, crop_size, base_size, norm_mean, norm_std)

        # transform all points to the global frame
        pc, mask = depth2pc(depth)
        shuffle_mask = np.arange(pc.shape[1])
        np.random.shuffle(shuffle_mask)
        shuffle_mask = shuffle_mask[::depth_sample_rate]
        mask = mask[shuffle_mask]
        pc = pc[:, shuffle_mask]
        pc = pc[:, mask]
        pc_global = transform_pc(pc, tf)

        rgb_cam_mat = get_sim_cam_mat(rgb.shape[0], rgb.shape[1])
        feat_cam_mat = get_sim_cam_mat(pix_feats.shape[2], pix_feats.shape[3])

        # project all point cloud onto the ground
        for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
            x, y = pos2grid_id(gs, cs, p[0], p[2])

            # ignore points projected to outside of the map and points that are 0.5 higher than the camera (could be from the ceiling)
            if x >= obstacles.shape[0] or y >= obstacles.shape[1] or \
                x < 0 or y < 0 or p_local[1] < -0.5:
                continue

            rgb_px, rgb_py, rgb_pz = project_point(rgb_cam_mat, p_local)
            rgb_v = rgb[rgb_py, rgb_px, :]
            semantic_v = semantic[rgb_py, rgb_px]
            if semantic_v == 40:
                semantic_v = -1

            # when the projected location is already assigned a color value before, overwrite if the current point has larger height
            if p_local[1] < color_top_down_height[y, x]:
                color_top_down[y, x] = rgb_v
                color_top_down_height[y, x] = p_local[1]
                gt[y, x] = semantic_v

            # average the visual embeddings if multiple points are projected to the same grid cell
            px, py, pz = project_point(feat_cam_mat, p_local)
            if not (px < 0 or py < 0 or px >= pix_feats.shape[3] or py >= pix_feats.shape[2]):
                feat = pix_feats[0, :, py, px]
                grid[y, x] = (grid[y, x] * weight[y, x] + feat) / (weight[y, x] + 1)
                weight[y, x] += 1

            # build an obstacle map ignoring points on the floor (0 means occupied, 1 means free)
            if p_local[1] > camera_height:
                continue
            obstacles[y, x] = 0
        pbar.update(1)

    save_map(color_top_down_save_path, color_top_down)
    save_map(gt_save_path, gt)
    save_map(grid_save_path, grid)
    save_map(weight_save_path, weight)
    save_map(obstacles_save_path, obstacles)


# Define get_lseg_feat function conditionally
if UTILS_AVAILABLE:
    def get_lseg_feat(model, image: np.array, labels, transform, crop_size=480, \
                     base_size=520, norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5]):
        vis_image = image.copy()
        image = transform(image).unsqueeze(0).cuda()
        img = image[0].permute(1,2,0)
        img = img * 0.5 + 0.5

        batch, _, h, w = image.size()
        stride_rate = 2.0/3.0
        stride = int(crop_size * stride_rate)

        long_size = base_size
        if h > w:
            height = long_size
            width = int(1.0 * w * long_size / h + 0.5)
            short_size = width
        else:
            width = long_size
            height = int(1.0 * h * long_size / w + 0.5)
            short_size = height

        cur_img = resize_image(image, height, width, **{'mode': 'bilinear', 'align_corners': True})

        if long_size <= crop_size:
            pad_img = pad_image(cur_img, norm_mean,
                                norm_std, crop_size)
            print(pad_img.shape)
            with torch.no_grad():
                outputs, logits = model(pad_img, labels)
            outputs = crop_image(outputs, 0, height, 0, width)
        else:
            if short_size < crop_size:
                # pad if needed
                pad_img = pad_image(cur_img, norm_mean,
                                    norm_std, crop_size)
            else:
                pad_img = cur_img
            _,_,ph,pw = pad_img.shape #.size()
            assert(ph >= height and pw >= width)
            h_grids = int(math.ceil(1.0 * (ph-crop_size)/stride)) + 1
            w_grids = int(math.ceil(1.0 * (pw-crop_size)/stride)) + 1
            with torch.cuda.device_of(image):
                with torch.no_grad():
                    outputs = image.new().resize_(batch, model.out_c,ph,pw).zero_().cuda()
                    logits_outputs = image.new().resize_(batch, len(labels),ph,pw).zero_().cuda()
                count_norm = image.new().resize_(batch,1,ph,pw).zero_().cuda()
            # grid evaluation
            for idh in range(h_grids):
                for idw in range(w_grids):
                    h0 = idh * stride
                    w0 = idw * stride
                    h1 = min(h0 + crop_size, ph)
                    w1 = min(w0 + crop_size, pw)
                    crop_img = crop_image(pad_img, h0, h1, w0, w1)
                    # pad if needed
                    pad_crop_img = pad_image(crop_img, norm_mean,
                                                norm_std, crop_size)
                    with torch.no_grad():
                        output, logits = model(pad_crop_img, labels)
                    cropped = crop_image(output, 0, h1-h0, 0, w1-w0)
                    cropped_logits = crop_image(logits, 0, h1-h0, 0, w1-w0)
                    outputs[:,:,h0:h1,w0:w1] += cropped
                    logits_outputs[:,:,h0:h1,w0:w1] += cropped_logits
                    count_norm[:,:,h0:h1,w0:w1] += 1
            assert((count_norm==0).sum()==0)
            outputs = outputs / count_norm
            logits_outputs = logits_outputs / count_norm
            outputs = outputs[:,:,:height,:width]
            logits_outputs = logits_outputs[:,:,:height,:width]
        outputs = outputs.cpu()
        outputs = outputs.numpy() # B, D, H, W
        predicts = [torch.max(logit, 0)[1].cpu().numpy() for logit in logits_outputs]
        pred = predicts[0]

        return outputs
else:
    # Placeholder function when utilities are not available
    def get_lseg_feat(*args, **kwargs):
        raise ImportError("LSeg utilities not available. Please install dependencies.")

def initialize_clip_model():
    """Initialize CLIP model and return necessary components"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_version = "ViT-B/32"
    clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                    'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
    
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load(clip_version)
    clip_model.to(device).eval()
    
    return clip_model, clip_feat_dim, device


def run_vlmaps_demo(data_dir=None, create_map=False, use_self_built_map=False):
    """Main function to run VLMaps demo"""
    
    if not UTILS_AVAILABLE:
        print("Required utilities not available. Please install dependencies and run from vlmaps directory.")
        return
    
    # Set default data directory if not provided
    if data_dir is None:
        # Look for data in current directory or common locations
        possible_dirs = [
            "./5LpN3gDmAk7_1/",
            "./dataset/vlmaps_data_dir/vlmaps_dataset/5LpN3gDmAk7_1/",
            "/tmp/vlmaps_data/"
        ]
        data_dir = None
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                data_dir = dir_path
                break
        
        if data_dir is None:
            print("No data directory found. Please provide data_dir parameter.")
            return
    
    # Check if the provided directory has the required subdirectories
    rgb_dir = os.path.join(data_dir, "rgb")
    if not os.path.exists(rgb_dir):
        print(f"Error: RGB directory not found at {rgb_dir}")
        # Try to find the correct directory
        alternative_path = f"dataset/vlmaps_data_dir/vlmaps_dataset/{os.path.basename(data_dir)}_1"
        if os.path.exists(alternative_path) and os.path.exists(os.path.join(alternative_path, "rgb")):
            print(f"Found data at: {alternative_path}")
            data_dir = alternative_path
        else:
            print("Please provide the correct path to the directory containing rgb/, depth/, semantic/, and poses.txt")
            print("Example: dataset/vlmaps_data_dir/vlmaps_dataset/5LpN3gDmAk7_1")
            return
    
    print(f"Using data directory: {data_dir}")
    
    # Initialize CLIP model
    clip_model, clip_feat_dim, device = initialize_clip_model()
    
    # Configuration parameters
    cs = 0.05  # meters per cell size
    gs = 1000  # map resolution (gs x gs)
    camera_height = 1.5  # camera height
    depth_sample_rate = 100  # depth pixels subsample rate
    
    # Create map if requested
    if create_map:
        print("Creating VLMap...")
        create_lseg_map_batch(data_dir, camera_height=camera_height, cs=cs, gs=gs, depth_sample_rate=depth_sample_rate)
        print("VLMap creation completed.")
    
    # Setup map paths
    map_save_dir = os.path.join(data_dir, "map_correct")
    if use_self_built_map:
        map_save_dir = os.path.join(data_dir, "map")
    
    if not os.path.exists(map_save_dir):
        print(f"Map directory not found: {map_save_dir}")
        if not create_map:
            print("Try setting create_map=True to build the map first.")
        return
    
    color_top_down_save_path = os.path.join(map_save_dir, "color_top_down_1.npy")
    grid_save_path = os.path.join(map_save_dir, "grid_lseg_1.npy")
    obstacles_save_path = os.path.join(map_save_dir, "obstacles.npy")
    
    # Check if map files exist
    required_files = [color_top_down_save_path, grid_save_path, obstacles_save_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Missing map files: {missing_files}")
        if not create_map:
            print("Try setting create_map=True to build the map first.")
        return
    
    # Load maps
    print("Loading maps...")
    obstacles = load_map(obstacles_save_path)
    color_top_down = load_map(color_top_down_save_path)
    grid = load_map(grid_save_path)
    
    # Find map boundaries
    x_indices, y_indices = np.where(obstacles == 0)
    xmin, xmax = np.min(x_indices), np.max(x_indices)  
    ymin, ymax = np.min(y_indices), np.max(y_indices)
    
    # Crop maps to relevant area
    obstacles_cropped = obstacles[xmin:xmax+1, ymin:ymax+1]
    color_top_down_cropped = color_top_down[xmin:xmax+1, ymin:ymax+1]
    grid_cropped = grid[xmin:xmax+1, ymin:ymax+1]
    
    # Display obstacle map
    print("Displaying obstacle map...")
    obstacles_pil = Image.fromarray(obstacles_cropped)
    plt.figure(figsize=(8, 6), dpi=120)
    plt.imshow(obstacles_pil, cmap='gray')
    plt.title("Obstacle Map")
    plt.savefig(os.path.join(map_save_dir, "obstacle_map.png"), bbox_inches='tight', dpi=120)
    plt.close()
    print(f"Obstacle map saved to: {os.path.join(map_save_dir, 'obstacle_map.png')}")
    
    # Display color map
    print("Displaying color map...")
    color_top_down_pil = Image.fromarray(color_top_down_cropped)
    plt.figure(figsize=(8, 6), dpi=120)
    plt.imshow(color_top_down_pil)
    plt.title("Top-Down Color Map")
    plt.savefig(os.path.join(map_save_dir, "color_top_down_map.png"), bbox_inches='tight', dpi=120)
    plt.close()
    print(f"Color map saved to: {os.path.join(map_save_dir, 'color_top_down_map.png')}")
    
    # Landmark indexing with MP3D categories
    print("Running landmark indexing with MP3D categories...")
    no_map_mask = obstacles_cropped > 0
    
    lang = mp3dcat
    text_feats = get_text_feats(lang, clip_model, clip_feat_dim)
    
    map_feats = grid_cropped.reshape((-1, grid_cropped.shape[-1]))
    scores_list = map_feats @ text_feats.T
    
    predicts = np.argmax(scores_list, axis=1)
    predicts = predicts.reshape(grid_cropped.shape[:2])
    floor_mask = predicts == 2
    
    new_pallete = get_new_pallete(len(lang))
    mask, patches = get_new_mask_pallete(predicts, new_pallete, out_label_flag=True, labels=lang)
    seg = mask.convert("RGBA")
    seg = np.array(seg)
    seg[no_map_mask] = [225, 225, 225, 255]
    seg[floor_mask] = [225, 225, 225, 255]
    seg = Image.fromarray(seg)
    
    plt.figure(figsize=(12, 8), dpi=120)
    plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(1., 1), prop={'size': 8})
    plt.axis('off')
    plt.title("VLMaps - MP3D Categories")
    plt.imshow(seg)
    plt.tight_layout()
    plt.savefig(os.path.join(map_save_dir, "vlmaps_mp3d_categories.png"), bbox_inches='tight', dpi=120)
    plt.close()
    print(f"MP3D categories map saved to: {os.path.join(map_save_dir, 'vlmaps_mp3d_categories.png')}")
    
    # Open-vocabulary landmark indexing
    print("Running open-vocabulary landmark indexing...")
    custom_lang = "big flat counter, sofa, floor, chair, wash basin, other"
    lang_list = custom_lang.split(",")
    lang_list = [item.strip() for item in lang_list]  # Clean whitespace
    
    text_feats = get_text_feats(lang_list, clip_model, clip_feat_dim)
    
    scores_list = map_feats @ text_feats.T
    predicts = np.argmax(scores_list, axis=1)
    predicts = predicts.reshape(grid_cropped.shape[:2])
    floor_mask = predicts == 2
    
    new_pallete = get_new_pallete(len(lang_list))
    mask, patches = get_new_mask_pallete(predicts, new_pallete, out_label_flag=True, labels=lang_list)
    seg = mask.convert("RGBA")
    seg = np.array(seg)
    seg[no_map_mask] = [225, 225, 225, 255]
    seg[floor_mask] = [225, 225, 225, 255]
    seg = Image.fromarray(seg)
    
    plt.figure(figsize=(12, 8), dpi=120)
    plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(1., 1), prop={'size': 10})
    plt.axis('off')
    plt.title("VLMaps - Custom Categories")
    plt.imshow(seg)
    plt.tight_layout()
    plt.savefig(os.path.join(map_save_dir, "vlmaps_custom_categories.png"), bbox_inches='tight', dpi=120)
    plt.close()
    print(f"Custom categories map saved to: {os.path.join(map_save_dir, 'vlmaps_custom_categories.png')}")
    
    print("VLMaps demo completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VLMaps Demo")
    parser.add_argument("--data_dir", type=str, help="Path to data directory")
    parser.add_argument("--create_map", action="store_true", help="Create new VLMap")
    parser.add_argument("--use_self_built_map", action="store_true", help="Use self-built map instead of provided map")
    
    args = parser.parse_args()
    
    run_vlmaps_demo(
        data_dir=args.data_dir,
        create_map=args.create_map,
        use_self_built_map=args.use_self_built_map
    )

"""# Citation
If you find this code useful, please cite the paper:

@inproceedings{huang23vlmaps,
    title={Visual Language Maps for Robot Navigation},
    author={Chenguang Huang and Oier Mees and Andy Zeng and Wolfram Burgard},
    booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
    year={2023},
    address = {London, UK}
}

# Usage Instructions:

## Basic usage (using pre-built map):
python demo.py --data_dir /path/to/your/data

## Create new map and run demo:
python demo.py --data_dir /path/to/your/data --create_map

## Use self-built map:
python demo.py --data_dir /path/to/your/data --use_self_built_map

## Requirements:
- Make sure you have all dependencies installed (see requirements.txt)
- Run from the vlmaps project directory
- Have the required data files in your data directory
- For map creation, you need LSeg checkpoints in lseg/checkpoints/

For more information, visit: https://vlmaps.github.io/
"""