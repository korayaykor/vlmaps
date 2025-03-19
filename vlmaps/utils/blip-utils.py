import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
from transformers import BlipProcessor, BlipModel, BlipForImageTextRetrieval

# Templates similar to CLIP's multiple_templates, adapted for BLIP
multiple_templates = [
    "There is {} in the scene.",
    "There is the {} in the scene.",
    "a photo of {} in the scene.",
    "a photo of the {} in the scene.",
    "a photo of one {} in the scene.",
    "I took a picture of {}.",
    "I took a picture of my {}.",
    "I took a picture of the {}.",
    "a photo of {}.",
    "a photo of my {}.",
    "a photo of the {}.",
    "a photo of one {}.",
    "a photo of many {}.",
    "a good photo of {}.",
    "a good photo of the {}.",
    "a photo of a nice {}.",
    "a photo of the nice {}.",
    "a photo of a cool {}.",
    "a photo of the cool {}.",
]

def get_blip_model(device="cuda"):
    """
    Initialize BLIP model and processor
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
    
    # Get feature model for embedding extraction
    feature_model = BlipModel.from_pretrained("Salesforce/blip-itm-base-coco")
    
    model.to(device)
    feature_model.to(device)
    
    return model, processor, feature_model

def get_text_features(texts, feature_model, processor, device="cuda", batch_size=64):
    """
    Extract text features using BLIP
    
    Args:
        texts: List of text strings
        feature_model: BLIP model for feature extraction
        processor: BLIP processor
        device: Device to run inference on
        batch_size: Batch size for processing
        
    Returns:
        Numpy array of text features
    """
    text_features = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
        
        # Move inputs to device
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(device)
        
        with torch.no_grad():
            outputs = feature_model(**inputs)
            # Extract text features from the text encoder output
            batch_features = outputs.text_embeds
            
            # Normalize features
            batch_features = F.normalize(batch_features, dim=-1)
            text_features.append(batch_features.cpu().numpy())
    
    # Concatenate all batches
    if text_features:
        text_features = np.concatenate(text_features, axis=0)
    else:
        text_features = np.array([])
    
    return text_features

def get_image_features(images, feature_model, processor, device="cuda", batch_size=64):
    """
    Extract image features using BLIP
    
    Args:
        images: List of numpy image arrays (RGB format)
        feature_model: BLIP model for feature extraction
        processor: BLIP processor
        device: Device to run inference on
        batch_size: Batch size for processing
        
    Returns:
        Numpy array of image features
    """
    image_features = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        processed_images = []
        
        for img in batch:
            # Convert numpy array to PIL Image if needed
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img.astype(np.uint8))
            
            processed_images.append(img)
        
        # Process images
        inputs = processor(images=processed_images, return_tensors="pt", padding=True)
        
        # Move inputs to device
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(device)
        
        with torch.no_grad():
            outputs = feature_model(**inputs)
            # Extract image features from the vision encoder output
            batch_features = outputs.image_embeds
            
            # Normalize features
            batch_features = F.normalize(batch_features, dim=-1)
            image_features.append(batch_features.cpu().numpy())
    
    # Concatenate all batches
    if image_features:
        image_features = np.concatenate(image_features, axis=0)
    else:
        image_features = np.array([])
    
    return image_features

def get_text_features_multiple_templates(texts, feature_model, processor, device="cuda", batch_size=64):
    """
    Apply multiple templates to texts and extract features, then average them
    
    Args:
        texts: List of text strings (concepts to be used in templates)
        feature_model: BLIP model for feature extraction
        processor: BLIP processor
        device: Device to run inference on
        batch_size: Batch size for processing
        
    Returns:
        Numpy array of averaged text features
    """
    # Apply templates to texts
    templated_texts = []
    for text in texts:
        for template in multiple_templates:
            templated_texts.append(template.format(text))
    
    # Get features for all templated texts
    all_features = get_text_features(templated_texts, feature_model, processor, device, batch_size)
    
    # Reshape to group by original text
    features_by_template = all_features.reshape(len(texts), len(multiple_templates), -1)
    
    # Average across templates
    averaged_features = np.mean(features_by_template, axis=1)
    
    return averaged_features

def compute_similarity(image_features, text_features):
    """
    Compute similarity scores between image and text features
    
    Args:
        image_features: Numpy array of image features
        text_features: Numpy array of text features
        
    Returns:
        Similarity scores matrix
    """
    # Ensure both feature sets are normalized
    image_features = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
    text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
    
    # Compute dot product similarity
    similarity = np.matmul(image_features, text_features.T)
    
    return similarity

def get_lseg_score(
    blip_model,
    blip_processor,
    blip_feature_model,
    landmarks,
    lseg_map,
    use_multiple_templates=False,
    add_other=True,
    device="cuda"
):
    """
    Compute similarity scores between LSEG map features and landmark text features
    
    Args:
        blip_model: BLIP model
        blip_processor: BLIP processor
        blip_feature_model: BLIP feature model
        landmarks: List of landmark text strings
        lseg_map: LSEG map features
        use_multiple_templates: Whether to use multiple templates
        add_other: Whether to add "other" category
        device: Device to run inference on
        
    Returns:
        Similarity scores matrix
    """
    landmarks_other = landmarks.copy()
    if add_other and "other" not in landmarks_other:
        landmarks_other.append("other")
    
    # Get text features
    if use_multiple_templates:
        text_features = get_text_features_multiple_templates(
            landmarks_other, blip_feature_model, blip_processor, device
        )
    else:
        text_features = get_text_features(
            landmarks_other, blip_feature_model, blip_processor, device
        )
    
    # Reshape LSeg map features for similarity computation
    map_features = lseg_map.reshape((-1, lseg_map.shape[-1]))
    
    # Compute similarity
    scores = compute_similarity(map_features, text_features)
    
    return scores

def match_text_to_imgs(text, images, blip_model, blip_processor, blip_feature_model, device="cuda"):
    """
    Match text to images using BLIP similarity
    
    Args:
        text: Text string
        images: List of images
        blip_model: BLIP model
        blip_processor: BLIP processor
        blip_feature_model: BLIP feature model
        device: Device to run inference on
        
    Returns:
        Similarity scores, image features, text features
    """
    # Get image features
    img_features = get_image_features(images, blip_feature_model, blip_processor, device)
    
    # Get text features
    text_features = get_text_features([text], blip_feature_model, blip_processor, device)
    
    # Compute similarity scores
    scores = compute_similarity(img_features, text_features)
    
    return scores.squeeze(), img_features, text_features

def get_nn_img(raw_imgs, text_features, img_features):
    """
    Get nearest neighbor images to text features
    
    Args:
        raw_imgs: List of original images
        text_features: Text features
        img_features: Image features
        
    Returns:
        Indices, images, and scores sorted by similarity
    """
    # Compute similarity scores
    scores = compute_similarity(img_features, text_features)
    scores = scores.squeeze()
    
    # Sort by similarity
    high_to_low_ids = np.argsort(scores)[::-1]
    high_to_low_imgs = [raw_imgs[i] for i in high_to_low_ids]
    high_to_low_scores = np.sort(scores)[::-1]
    
    return high_to_low_ids, high_to_low_imgs, high_to_low_scores
