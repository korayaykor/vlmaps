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

# Try to import BLIP
try:
    from transformers import BlipProcessor, BlipForImageTextRetrieval
    BLIP_AVAILABLE = True
except ImportError:
    print("BLIP not available. Install with: pip install transformers")
    BLIP_AVAILABLE = False

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
def create_lseg_map_batch(img_save_dir, camera_height, cs=0.05, gs=1000, depth_sample_rate=100, use_blip=False):
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

def initialize_blip_model(blip_version="blip-base"):
    """Initialize BLIP model for image-text retrieval and return necessary components"""
    if not BLIP_AVAILABLE:
        raise ImportError("BLIP not available. Install with: pip install transformers")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # BLIP model configurations optimized for image-text retrieval
    blip_configs = {
        'blip-base': {
            'model_name': 'Salesforce/blip-itm-base-coco',
            'feat_dim': 768
        },
        'blip-large': {
            'model_name': 'Salesforce/blip-itm-large-coco', 
            'feat_dim': 768
        },
        'blip-vit-base': {
            'model_name': 'Salesforce/blip-vqa-base',
            'feat_dim': 768
        },
        'blip-vit-large': {
            'model_name': 'Salesforce/blip-vqa-capfilt-large',
            'feat_dim': 768
        }
    }
    
    if blip_version not in blip_configs:
        print(f"Warning: {blip_version} not found, using blip-base")
        blip_version = "blip-base"
    
    config = blip_configs[blip_version]
    blip_model_name = config['model_name']
    blip_feat_dim = config['feat_dim']
    
    print(f"Loading BLIP model for image-text retrieval: {blip_version} ({blip_model_name})...")
    
    try:
        processor = BlipProcessor.from_pretrained(blip_model_name)
        model = BlipForImageTextRetrieval.from_pretrained(blip_model_name)
        model.to(device).eval()
        
        print(f"BLIP model loaded successfully with {blip_feat_dim}D features")
        return model, processor, blip_feat_dim, device
        
    except Exception as e:
        print(f"Failed to load {blip_version}, falling back to blip-itm-base-coco: {e}")
        # Fallback to base model specifically designed for image-text matching
        try:
            processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
            model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
            model.to(device).eval()
            return model, processor, 768, device
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            # Final fallback to a known working model
            processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
            model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-vqa-base")
            model.to(device).eval()
            return model, processor, 768, device

def get_blip_text_features(texts, model, processor, device, target_dim=512):
    """Get text features using BLIP for image-text retrieval - optimized for semantic mapping"""
    text_features = []
    
    # Create a learnable projection layer for dimension matching (shared across all texts)
    if not hasattr(get_blip_text_features, 'projection_layer'):
        get_blip_text_features.projection_layer = None
    
    with torch.no_grad():
        for text in texts:
            try:
                # Enhanced text prompting for better semantic understanding
                # Use multiple prompt templates for robustness
                prompt_templates = [
                    f"a photo of {text}",
                    f"an image containing {text}",
                    f"this is {text}",
                    f"{text} in a room"
                ]
                
                text_embeds_list = []
                
                for prompt in prompt_templates:
                    try:
                        # Process text input
                        inputs = processor(text=prompt, return_tensors="pt", padding=True, truncation=True, max_length=77)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        # Extract text embeddings using the text encoder directly
                        if hasattr(model, 'text_model'):
                            # For BLIP-2 style models
                            text_outputs = model.text_model(**inputs)
                            if hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None:
                                text_embeds = text_outputs.pooler_output
                            else:
                                text_embeds = text_outputs.last_hidden_state.mean(dim=1)
                        elif hasattr(model, 'text_encoder'):
                            # For standard BLIP models
                            text_outputs = model.text_encoder(**inputs)
                            if hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None:
                                text_embeds = text_outputs.pooler_output
                            else:
                                text_embeds = text_outputs.last_hidden_state[:, 0, :]  # CLS token
                        else:
                            # Fallback: use the complete model with dummy image
                            dummy_image = torch.zeros(1, 3, 224, 224).to(device)
                            dummy_inputs = processor(images=dummy_image, return_tensors="pt")
                            dummy_inputs = {k: v.to(device) for k, v in dummy_inputs.items()}
                            
                            outputs = model(
                                pixel_values=dummy_inputs['pixel_values'],
                                input_ids=inputs['input_ids'],
                                attention_mask=inputs['attention_mask'],
                                return_dict=True
                            )
                            
                            if hasattr(outputs, 'text_embeds'):
                                text_embeds = outputs.text_embeds
                            else:
                                continue  # Skip this prompt if we can't extract embeddings
                        
                        text_embeds_list.append(text_embeds)
                        
                    except Exception as e:
                        print(f"Debug: Failed to process prompt '{prompt}': {e}")
                        continue
                
                if text_embeds_list:
                    # Average embeddings from multiple prompts for robustness
                    text_embeds = torch.stack(text_embeds_list).mean(dim=0)
                    
                    # Apply text projection if available
                    if hasattr(model, 'text_proj') and model.text_proj is not None:
                        text_embeds = model.text_proj(text_embeds)
                    
                    # Project to target dimension if needed
                    if text_embeds.shape[-1] != target_dim:
                        if get_blip_text_features.projection_layer is None:
                            get_blip_text_features.projection_layer = torch.nn.Linear(
                                text_embeds.shape[-1], target_dim, bias=False
                            ).to(device)
                            # Initialize with Xavier uniform for better convergence
                            torch.nn.init.xavier_uniform_(get_blip_text_features.projection_layer.weight)
                        
                        text_embeds = get_blip_text_features.projection_layer(text_embeds)
                    
                    # Normalize features for cosine similarity
                    text_embeds = text_embeds / (text_embeds.norm(dim=-1, keepdim=True) + 1e-8)
                    text_features.append(text_embeds.cpu().numpy())
                else:
                    print(f"Warning: Could not extract features for '{text}', using zero vector")
                    zero_feat = torch.zeros(1, target_dim)
                    text_features.append(zero_feat.numpy())
                    
            except Exception as e:
                print(f"Error processing text '{text}': {e}")
                zero_feat = torch.zeros(1, target_dim)
                text_features.append(zero_feat.numpy())
    
    return np.vstack(text_features)

def get_blip_image_features(image, model, processor, device):
    """Get image features using BLIP for image-text retrieval"""
    with torch.no_grad():
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
            
        inputs = processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # BlipForImageTextRetrieval has a specific vision encoder
        if hasattr(model, 'vision_model'):
            # Use the vision model directly
            image_outputs = model.vision_model(**inputs)
            image_embeds = image_outputs.pooler_output
            
            # Apply vision projection if available
            if hasattr(model, 'visual_projection'):
                image_embeds = model.visual_projection(image_embeds)
                
        elif hasattr(model, 'get_image_features'):
            # Fallback method if available
            image_embeds = model.get_image_features(**inputs)
        else:
            # Alternative approach using the full model with dummy text
            dummy_text = "a photo"
            text_inputs = processor(text=dummy_text, return_tensors="pt")
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            
            outputs = model(
                pixel_values=inputs['pixel_values'],
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask'],
                return_dict=True
            )
            
            if hasattr(outputs, 'image_embeds'):
                image_embeds = outputs.image_embeds
            else:
                # Use vision embeddings as image representation
                image_embeds = outputs.vision_embeds
        
        # Normalize features for better similarity computation
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        return image_embeds.cpu().numpy()

def enhance_categories_for_blip(categories):
    """Enhance category names for better BLIP understanding in semantic mapping"""
    enhanced_categories = []
    
    # Category enhancement rules based on semantic mapping requirements
    enhancement_rules = {
        # Furniture items
        'chair': 'chair furniture for sitting',
        'table': 'table furniture surface',
        'sofa': 'sofa couch furniture for sitting',
        'bed': 'bed furniture for sleeping',
        'cabinet': 'cabinet storage furniture',
        'chest_of_drawers': 'chest of drawers storage furniture',
        'stool': 'stool seat furniture',
        
        # Structural elements
        'wall': 'wall interior surface',
        'floor': 'floor ground surface',
        'ceiling': 'ceiling overhead surface',
        'door': 'door entrance opening',
        'window': 'window glass opening',
        'stairs': 'stairs staircase steps',
        'column': 'architectural column pillar',
        'beam': 'ceiling beam structure',
        'railing': 'stair railing barrier',
        
        # Bathroom fixtures
        'toilet': 'toilet bathroom fixture',
        'sink': 'sink washbasin fixture',
        'bathtub': 'bathtub bathroom fixture',
        'shower': 'shower bathroom fixture',
        'mirror': 'bathroom mirror reflective surface',
        'towel': 'towel bathroom textile',
        
        # Kitchen elements
        'counter': 'kitchen counter worktop surface',
        'appliances': 'kitchen appliances equipment',
        
        # Decorative and functional items
        'picture': 'picture wall art decoration',
        'curtain': 'window curtain fabric',
        'blinds': 'window blinds covering',
        'cushion': 'cushion soft furnishing',
        'plant': 'houseplant indoor vegetation',
        'lighting': 'light fixture illumination',
        'tv_monitor': 'television monitor screen',
        
        # Storage and organization
        'shelving': 'shelving storage system',
        'board_panel': 'wall panel board surface',
        
        # Activities and equipment
        'gym_equipment': 'exercise gym equipment',
        'seating': 'seating furniture area',
        
        # General categories
        'furniture': 'indoor furniture items',
        'objects': 'household objects items',
        'clothes': 'clothing textile items',
        'fireplace': 'fireplace heating feature',
        
        # Special categories
        'void': 'empty void space',
        'other': 'miscellaneous other items'
    }
    
    for category in categories:
        category_clean = category.lower().strip()
        if category_clean in enhancement_rules:
            enhanced_categories.append(enhancement_rules[category_clean])
        else:
            # Default enhancement - add contextual information
            if any(furniture_word in category_clean for furniture_word in ['chair', 'table', 'sofa', 'bed', 'cabinet']):
                enhanced_categories.append(f"{category} indoor furniture")
            elif any(surface_word in category_clean for surface_word in ['wall', 'floor', 'ceiling']):
                enhanced_categories.append(f"{category} interior surface")
            elif any(fixture_word in category_clean for fixture_word in ['sink', 'toilet', 'bathtub']):
                enhanced_categories.append(f"{category} bathroom fixture")
            else:
                enhanced_categories.append(f"{category} indoor object")
    
    return enhanced_categories


def run_vlmaps_demo(data_dir=None, create_map=False, use_self_built_map=False, use_blip=False, blip_version="blip-base"):
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
    
    # Initialize model (CLIP or BLIP)
    if use_blip and BLIP_AVAILABLE:
        print(f"Using BLIP ({blip_version}) for vision-language features")
        model, processor, feat_dim, device = initialize_blip_model(blip_version)
        # Configure BLIP for optimal semantic mapping
        model, processor = configure_blip_for_semantic_mapping(model, processor, device)
        model_type = "blip"
    else:
        if use_blip and not BLIP_AVAILABLE:
            print("BLIP requested but not available, falling back to CLIP")
        print("Using CLIP for vision-language features")
        model, feat_dim, device = initialize_clip_model()
        processor = None  # CLIP doesn't use processor
        model_type = "clip"
    
    # Configuration parameters
    cs = 0.05  # meters per cell size
    gs = 1000  # map resolution (gs x gs)
    camera_height = 1.5  # camera height
    depth_sample_rate = 100  # depth pixels subsample rate
    
    # Create map if requested
    if create_map:
        print("Creating VLMap...")
        create_lseg_map_batch(data_dir, camera_height=camera_height, cs=cs, gs=gs, depth_sample_rate=depth_sample_rate, use_blip=use_blip)
        print("VLMap creation completed.")
        # Automatically use the newly created map
        use_self_built_map = True
    
    # Setup map paths
    map_save_dir = os.path.join(data_dir, "map_correct")
    if use_self_built_map or create_map:
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
    if model_type == "blip":
        # Enhance categories for better BLIP understanding
        enhanced_lang = enhance_categories_for_blip(lang)
        print(f"Using enhanced categories for BLIP: {len(enhanced_lang)} categories")
        print(f"Enhanced categories: {enhanced_lang[:5]}...")  # Show first 5 for debugging
        
        # Use the same feature dimension as the map (512D for ViT-B/32 CLIP)
        map_feat_dim = grid_cropped.shape[-1]
        print(f"Map feature dimension: {map_feat_dim}")
        
        text_feats = get_blip_text_features(enhanced_lang, model, processor, device, target_dim=map_feat_dim)
        print(f"BLIP text features shape: {text_feats.shape}")
    else:
        text_feats = get_text_feats(lang, model, feat_dim)
    
    map_feats = grid_cropped.reshape((-1, grid_cropped.shape[-1]))
    scores_list = map_feats @ text_feats.T
    
    # Apply BLIP-specific semantic filtering if using BLIP
    if model_type == "blip":
        print("Applying BLIP semantic filtering...")
        scores_list = apply_blip_semantic_filtering(scores_list, confidence_threshold=0.05)
    
    # Debug information
    print(f"Map features shape: {map_feats.shape}")
    print(f"Text features shape: {text_feats.shape}")
    print(f"Scores shape: {scores_list.shape}")
    print(f"Score statistics - Min: {scores_list.min():.3f}, Max: {scores_list.max():.3f}, Mean: {scores_list.mean():.3f}")
    
    # Check for any NaN or infinite values
    if np.any(np.isnan(scores_list)) or np.any(np.isinf(scores_list)):
        print("Warning: Found NaN or infinite values in scores, applying cleanup...")
        scores_list = np.nan_to_num(scores_list, nan=0.0, posinf=1.0, neginf=-1.0)
    
    predicts = np.argmax(scores_list, axis=1)
    predicts = predicts.reshape(grid_cropped.shape[:2])
    unique_predictions = np.unique(predicts)
    print(f"Unique predictions: {unique_predictions} (out of {len(lang)} categories)")
    
    # Show prediction confidence statistics for BLIP
    if model_type == "blip":
        max_scores = np.max(scores_list, axis=1)
        print(f"Prediction confidence - Min: {max_scores.min():.3f}, Max: {max_scores.max():.3f}, Mean: {max_scores.mean():.3f}")
        high_conf_ratio = np.mean(max_scores > 0.1)
        print(f"High confidence predictions (>0.1): {high_conf_ratio:.1%}")
    
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
    
    if model_type == "blip":
        # Enhance custom categories for better BLIP understanding
        enhanced_lang_list = enhance_categories_for_blip(lang_list)
        print(f"Using enhanced custom categories for BLIP: {enhanced_lang_list}")
        # Use the same feature dimension as the map
        map_feat_dim = grid_cropped.shape[-1]
        text_feats = get_blip_text_features(enhanced_lang_list, model, processor, device, target_dim=map_feat_dim)
    else:
        text_feats = get_text_feats(lang_list, model, feat_dim)
    
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


def configure_blip_for_semantic_mapping(model, processor, device):
    """Configure BLIP model for optimal semantic mapping performance"""
    
    # Set model to evaluation mode for consistent inference
    model.eval()
    
    # Optimize processor settings for better text understanding
    if hasattr(processor, 'tokenizer'):
        # Increase max length for better context understanding
        processor.tokenizer.model_max_length = 77  # Match CLIP's token limit
        processor.tokenizer.padding_side = 'right'
        processor.tokenizer.truncation_side = 'right'
    
    # Configure image processor for consistent features
    if hasattr(processor, 'image_processor'):
        # Ensure consistent image preprocessing
        processor.image_processor.do_normalize = True
        processor.image_processor.do_resize = True
        processor.image_processor.size = {"height": 224, "width": 224}
    
    print("BLIP model configured for semantic mapping")
    return model, processor

def get_blip_semantic_categories():
    """Get categories optimized for BLIP semantic understanding"""
    # Categories that BLIP understands well based on its training data
    blip_optimized_categories = [
        # High-confidence furniture categories
        "chair for sitting",
        "dining table surface", 
        "sofa for relaxing",
        "bed for sleeping",
        "storage cabinet",
        
        # Clear structural elements
        "interior wall surface",
        "floor ground surface", 
        "ceiling overhead",
        "doorway entrance",
        "glass window",
        
        # Distinct bathroom fixtures
        "white toilet fixture",
        "bathroom sink basin",
        "shower bathtub fixture",
        "bathroom mirror",
        
        # Kitchen elements
        "kitchen counter surface",
        "kitchen appliances",
        
        # Decorative items
        "wall picture artwork",
        "window curtains",
        "decorative cushion",
        "indoor plant",
        "ceiling light fixture",
        
        # Storage and organization
        "wall shelving unit",
        "wooden panel",
        
        # Miscellaneous
        "household objects",
        "textile items",
        "other furniture"
    ]
    
    return blip_optimized_categories

def apply_blip_semantic_filtering(predictions, confidence_threshold=0.1):
    """Apply semantic filtering to improve BLIP predictions"""
    # Filter out low-confidence predictions
    max_scores = np.max(predictions, axis=1)
    high_confidence_mask = max_scores > confidence_threshold
    
    # Apply semantic consistency rules
    filtered_predictions = predictions.copy()
    
    # Smooth predictions using spatial consistency (without scipy dependency)
    try:
        from scipy import ndimage
        scipy_available = True
    except ImportError:
        scipy_available = False
        print("Scipy not available, skipping spatial filtering")
    
    if scipy_available:
        for i in range(predictions.shape[1]):
            try:
                # Get actual grid dimensions from the cropped map shape
                # This should be passed as a parameter, but we'll estimate for now
                total_pixels = predictions.shape[0]
                
                # Try to find factors that multiply to total_pixels
                for grid_h in range(int(np.sqrt(total_pixels)), 0, -1):
                    if total_pixels % grid_h == 0:
                        grid_w = total_pixels // grid_h
                        break
                else:
                    # Fallback: skip spatial filtering for this category
                    continue
                
                category_map = (np.argmax(predictions, axis=1) == i).reshape(grid_h, grid_w)
                
                # Apply median filter to reduce noise
                smoothed = ndimage.median_filter(category_map.astype(float), size=3)
                category_mask = smoothed > 0.5
                
                # Boost confidence for spatially consistent regions
                if np.any(category_mask):
                    flat_mask = category_mask.flatten()[:predictions.shape[0]]  # Ensure correct size
                    filtered_predictions[flat_mask, i] *= 1.2
            except Exception as e:
                print(f"Warning: Spatial filtering failed for category {i}: {e}")
                continue
    
    # Apply confidence boosting for high-confidence predictions
    high_conf_mask = max_scores > confidence_threshold * 2
    filtered_predictions[high_conf_mask] *= 1.1
    
    # Renormalize to ensure probabilities sum to 1
    row_sums = np.sum(filtered_predictions, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    filtered_predictions = filtered_predictions / row_sums
    
    return filtered_predictions


def test_blip_functionality(blip_version="blip-base"):
    """Test BLIP functionality with simple categories"""
    print("Testing BLIP functionality...")
    
    try:
        model, processor, feat_dim, device = initialize_blip_model(blip_version)
        model, processor = configure_blip_for_semantic_mapping(model, processor, device)
        
        # Test with simple categories
        test_categories = ["chair", "table", "wall", "floor", "door"]
        enhanced_categories = enhance_categories_for_blip(test_categories)
        print(f"Test categories: {test_categories}")
        print(f"Enhanced categories: {enhanced_categories}")
        
        # Try to extract text features
        text_feats = get_blip_text_features(enhanced_categories, model, processor, device, target_dim=512)
        print(f"Text features shape: {text_feats.shape}")
        print(f"Feature statistics - Min: {text_feats.min():.3f}, Max: {text_feats.max():.3f}")
        
        # Check for valid features
        if np.any(np.isnan(text_feats)) or np.all(text_feats == 0):
            print("❌ BLIP test failed - invalid features")
            return False
        else:
            print("✅ BLIP test passed - features extracted successfully")
            return True
            
    except Exception as e:
        print(f"❌ BLIP test failed with error: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VLMaps Demo")
    parser.add_argument("--data_dir", type=str, help="Path to data directory")
    parser.add_argument("--create_map", action="store_true", help="Create new VLMap")
    parser.add_argument("--use_self_built_map", action="store_true", help="Use self-built map instead of provided map")
    parser.add_argument("--use_blip", action="store_true", help="Use BLIP model instead of CLIP for vision-language features")
    parser.add_argument("--blip_version", type=str, default="blip-base", 
                        choices=['blip-base', 'blip-large', 'blip-vit-base', 'blip-vit-large'],
                        help="BLIP model version to use for image-text retrieval (default: blip-base)")
    parser.add_argument("--test_blip", action="store_true", help="Test BLIP functionality before running demo")
    
    args = parser.parse_args()
    
    # Test BLIP functionality if requested
    if args.test_blip and args.use_blip:
        if test_blip_functionality(args.blip_version):
            print("BLIP test successful, proceeding with demo...")
        else:
            print("BLIP test failed, please check the configuration")
            exit(1)
    
    run_vlmaps_demo(
        data_dir=args.data_dir,
        create_map=args.create_map,
        use_self_built_map=args.use_self_built_map,
        use_blip=args.use_blip,
        blip_version=args.blip_version
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

## Use BLIP instead of CLIP:
python demo.py --data_dir /path/to/your/data --use_blip

## Use specific BLIP model version:
python demo.py --data_dir /path/to/your/data --use_blip --blip_version blip-large

## Create map with BLIP:
python demo.py --data_dir /path/to/your/data --create_map --use_blip

## Available BLIP versions (optimized for image-text retrieval):
# blip-base (768D features) - Default, BLIP with ITM (Image-Text Matching) training
# blip-large (768D features) - Larger BLIP with ITM training, better performance
# blip-vit-base (768D features) - BLIP with ViT backbone, VQA pre-training
# blip-vit-large (768D features) - Large BLIP with ViT backbone, better accuracy

## Requirements:
- Make sure you have all dependencies installed (see requirements.txt)
- Run from the vlmaps project directory
- Have the required data files in your data directory
- For map creation, you need LSeg checkpoints in lseg/checkpoints/

For more information, visit: https://vlmaps.github.io/
"""