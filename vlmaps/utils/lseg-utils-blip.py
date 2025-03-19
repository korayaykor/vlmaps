import math
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import BlipProcessor, BlipModel

from vlmaps.utils.mapping_utils import get_new_pallete, get_new_mask_pallete
from vlmaps.lseg.additional_utils.models import resize_image, pad_image, crop_image


def get_lseg_blip_feat(
    model,
    image: np.ndarray,
    labels,
    processor,
    device,
    crop_size=480,
    base_size=520,
    norm_mean=[0.5, 0.5, 0.5],
    norm_std=[0.5, 0.5, 0.5],
    vis=False,
):
    """
    Get LSeg features using BLIP model instead of CLIP
    """
    vis_image = image.copy()
    
    # Convert numpy array to PIL image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Process the image with BLIP processor
    inputs = processor(images=image, return_tensors="pt")
    
    # Move inputs to device
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].to(device)
    
    batch, c, h, w = inputs['pixel_values'].shape
    stride_rate = 2.0 / 3.0
    stride = int(crop_size * stride_rate)

    # Handle image size 
    long_size = base_size
    if h > w:
        height = long_size
        width = int(1.0 * w * long_size / h + 0.5)
        short_size = width
    else:
        width = long_size
        height = int(1.0 * h * long_size / w + 0.5)
        short_size = height

    # Resize image
    cur_img = resize_image(inputs['pixel_values'], height, width, **{"mode": "bilinear", "align_corners": True})
    
    # Handle cropping for large images
    if long_size <= crop_size:
        pad_img = pad_image(cur_img, norm_mean, norm_std, crop_size)
        with torch.no_grad():
            # Get features from BLIP
            outputs = model(pixel_values=pad_img, input_ids=None, return_dict=True)
            feats = outputs.vision_model_output.last_hidden_state
            
            # Reshape to match LSeg format
            b, hw, c = feats.shape
            h = w = int(math.sqrt(hw))
            feats = feats.reshape(b, h, w, c).permute(0, 3, 1, 2)
            
        outputs = crop_image(feats, 0, height, 0, width)
    else:
        if short_size < crop_size:
            # Pad if needed
            pad_img = pad_image(cur_img, norm_mean, norm_std, crop_size)
        else:
            pad_img = cur_img
            
        _, _, ph, pw = pad_img.shape
        assert ph >= height and pw >= width
        
        # Calculate grids for larger images
        h_grids = int(math.ceil(1.0 * (ph - crop_size) / stride)) + 1
        w_grids = int(math.ceil(1.0 * (pw - crop_size) / stride)) + 1
        
        with torch.cuda.device_of(inputs['pixel_values']):
            with torch.no_grad():
                # Create empty tensors for outputs
                feat_dim = model.config.hidden_size
                outputs = torch.zeros((batch, feat_dim, ph, pw), device=device)
                count_norm = torch.zeros((batch, 1, ph, pw), device=device)
                
            # Process image in grid-wise manner
            for idh in range(h_grids):
                for idw in range(w_grids):
                    h0 = idh * stride
                    w0 = idw * stride
                    h1 = min(h0 + crop_size, ph)
                    w1 = min(w0 + crop_size, pw)
                    
                    # Crop the image
                    crop_img = crop_image(pad_img, h0, h1, w0, w1)
                    
                    # Pad if needed
                    pad_crop_img = pad_image(crop_img, norm_mean, norm_std, crop_size)
                    
                    with torch.no_grad():
                        # Get features from BLIP
                        out = model(pixel_values=pad_crop_img, input_ids=None, return_dict=True)
                        feats = out.vision_model_output.last_hidden_state
                        
                        # Reshape to match LSeg format
                        b, hw, c = feats.shape
                        h = w = int(math.sqrt(hw))
                        feats = feats.reshape(b, h, w, c).permute(0, 3, 1, 2)
                        
                    # Crop and add to output
                    cropped = crop_image(feats, 0, h1 - h0, 0, w1 - w0)
                    outputs[:, :, h0:h1, w0:w1] += cropped
                    count_norm[:, :, h0:h1, w0:w1] += 1
                    
            assert (count_norm == 0).sum() == 0
            outputs = outputs / count_norm
            outputs = outputs[:, :, :height, :width]
            
    # Normalize and convert to numpy
    outputs = outputs.cpu().numpy()
    
    if vis:
        # Visualize the features
        if isinstance(labels, list) and len(labels) > 0:
            # Process text with BLIP
            inputs_text = processor(text=labels, return_tensors="pt", padding=True)
            for key in inputs_text:
                if isinstance(inputs_text[key], torch.Tensor):
                    inputs_text[key] = inputs_text[key].to(device)
                    
            with torch.no_grad():
                text_outputs = model(input_ids=inputs_text['input_ids'], attention_mask=inputs_text['attention_mask'], return_dict=True)
                text_feats = text_outputs.text_embeds
                
                # Normalize image and text features
                image_feats = torch.from_numpy(outputs).to(device)
                image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
                text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)
                
                # Compute similarity
                similarity = torch.einsum('bchw,nc->bnhw', image_feats, text_feats)
                
                # Get class predictions
                pred = similarity.argmax(dim=1)[0].cpu().numpy()
                
                # Visualize
                new_palette = get_new_pallete(len(labels))
                mask, patches = get_new_mask_pallete(pred, new_palette, out_label_flag=True, labels=labels)
                seg = mask.convert("RGBA")
                
                cv2.imshow("image", vis_image[:, :, [2, 1, 0]])
                cv2.waitKey()
                
                fig = plt.figure()
                plt.imshow(seg)
                plt.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.0, 1), prop={"size": 20})
                plt.axis("off")
                plt.tight_layout()
                plt.show()

    return outputs
