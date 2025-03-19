import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BlipProcessor, BlipModel

from .lseg_blocks import FeatureFusionBlock, Interpolate, _make_encoder, FeatureFusionBlock_custom
import numpy as np


class depthwise_clipseg_conv(nn.Module):
    def __init__(self):
        super(depthwise_clipseg_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    
    def depthwise_clipseg(self, x, channels):
        x = torch.cat([self.depthwise(x[:, i].unsqueeze(1)) for i in range(channels)], dim=1)
        return x

    def forward(self, x):
        channels = x.shape[1]
        out = self.depthwise_clipseg(x, channels)
        return out


class depthwise_conv(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # support for 4D tensor with NCHW
        C, H, W = x.shape[1:]
        x = x.reshape(-1, 1, H, W)
        x = self.depthwise(x)
        x = x.view(-1, C, H, W)
        return x


class depthwise_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(depthwise_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x, act=True):
        x = self.depthwise(x)
        if act:
            x = self.activation(x)
        return x


class bottleneck_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(bottleneck_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()


    def forward(self, x, act=True):
        sum_layer = x.max(dim=1, keepdim=True)[0]
        x = self.depthwise(x)
        x = x + sum_layer
        if act:
            x = self.activation(x)
        return x

class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.
        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)

def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        activation=nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )

class LSeg_BLIP(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="clip_vitl16_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs,
    ):
        super(LSeg_BLIP, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "clip_vitl16_384": [5, 11, 17, 23],
            "clipRN50x16_vitl16_384": [5, 11, 17, 23],
            "clip_vitb32_384": [2, 5, 8, 11],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        # Initialize BLIP model and processor for text encoding
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
        self.blip_model = BlipModel.from_pretrained("Salesforce/blip-itm-base-coco")
        self.blip_model.to(device)
        
        # Get BLIP feature dimension
        self.out_c = self.blip_model.config.hidden_size
        
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

        self.arch_option = kwargs["arch_option"]
        if self.arch_option == 1:
            self.scratch.head_block = bottleneck_block(activation=kwargs["activation"])
            self.block_depth = kwargs['block_depth']
        elif self.arch_option == 2:
            self.scratch.head_block = depthwise_block(activation=kwargs["activation"])
            self.block_depth = kwargs['block_depth']

        self.scratch.output_conv = head
        
    def forward(self, x, labelset=''):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = self._forward_backbone(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # Get text features using BLIP
        device = x.device
        if labelset == '':
            text = self.labels  # Default labels if no labelset provided
        else:
            if isinstance(labelset, list):
                text = labelset
            else:
                text = [labelset]
        
        # Process text with BLIP
        inputs = self.blip_processor(text=text, return_tensors="pt", padding=True)
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(device)
                
        with torch.no_grad():
            outputs = self.blip_model(**inputs)
            text_features = outputs.text_embeds
            text_features = F.normalize(text_features, dim=-1)

        # Get image features
        image_features = self.scratch.head1(path_1)
        imshape = image_features.shape
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, self.out_c)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        logits_per_image = torch.matmul(image_features, text_features.t())
        
        # Reshape back to spatial dimensions
        out = logits_per_image.view(imshape[0], imshape[2], imshape[3], -1).permute(0, 3, 1, 2)

        if self.arch_option in [1, 2]:
            for _ in range(self.block_depth - 1):
                out = self.scratch.head_block(out)
            out = self.scratch.head_block(out, False)

        out = self.scratch.output_conv(out)
            
        return out
    
    def _forward_backbone(self, backbone, x):
        """Forward pass through the backbone"""
        b, c, h, w = x.shape
        
        # Use regular backbone forward method
        activation_dict = {}
        
        def hook_fn(module, input, output, name):
            activation_dict[name] = output
            
        handles = []
        for i, block in enumerate(backbone.blocks):
            if i in [5, 11, 17, 23]:  # Assuming these are the hook points
                handle = block.register_forward_hook(
                    lambda module, input, output, name=i: hook_fn(module, input, output, name)
                )
                handles.append(handle)
                
        _ = backbone(x)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
            
        # Get activations
        layer_1 = activation_dict.get(5, None)
        layer_2 = activation_dict.get(11, None)
        layer_3 = activation_dict.get(17, None)
        layer_4 = activation_dict.get(23, None)
        
        return layer_1, layer_2, layer_3, layer_4


class LSegBLIPNet(LSeg_BLIP):
    """Network for semantic segmentation with BLIP."""
    def __init__(self, labels, path=None, scale_factor=0.5, crop_size=480, **kwargs):
        features = kwargs["features"] if "features" in kwargs else 256
        kwargs["use_bn"] = True

        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.labels = labels

        head = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)


class LSegEncBLIP(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="clip_vitl16_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs,
    ):
        super(LSegEncBLIP, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "clip_vitl16_384": [5, 11, 17, 23],
            "clipRN50x16_vitl16_384": [5, 11, 17, 23],
            "clip_vitb32_384": [2, 5, 8, 11],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        # Initialize BLIP model and processor for text encoding
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
        self.blip_model = BlipModel.from_pretrained("Salesforce/blip-itm-base-coco")
        self.blip_model.to(device)
        
        # Get BLIP feature dimension
        self.out_c = self.blip_model.config.hidden_size
        
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

        self.arch_option = kwargs["arch_option"]
        if self.arch_option == 1:
            self.scratch.head_block = bottleneck_block(activation=kwargs["activation"])
            self.block_depth = kwargs['block_depth']
        elif self.arch_option == 2:
            self.scratch.head_block = depthwise_block(activation=kwargs["activation"])
            self.block_depth = kwargs['block_depth']

        self.scratch.output_conv = head
        
    def forward(self, x, labelset=''):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = self._forward_backbone(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # Get text features using BLIP
        device = x.device
        if labelset == '':
            text = self.labels  # Default labels if no labelset provided
        else:
            if isinstance(labelset, list):
                text = labelset
            else:
                text = [labelset]
        
        # Process text with BLIP
        inputs = self.blip_processor(text=text, return_tensors="pt", padding=True)
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(device)
                
        with torch.no_grad():
            outputs = self.blip_model(**inputs)
            text_features = outputs.text_embeds
            text_features = F.normalize(text_features, dim=-1)

        # Get image features
        image_features = self.scratch.head1(path_1)
        imshape = image_features.shape
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, self.out_c)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Reshape pixel encoding for output
        pixel_encoding = image_features.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0, 3, 1, 2)
        
        # Compute similarity
        logits_per_image = torch.matmul(image_features, text_features.t())
        
        # Reshape back to spatial dimensions
        out = logits_per_image.view(imshape[0], imshape[2], imshape[3], -1).permute(0, 3, 1, 2)

        # Apply output conv to both outputs
        pixel_encoding = self.scratch.output_conv(pixel_encoding)
        out = self.scratch.output_conv(out)
            
        return pixel_encoding, out
    
    def _forward_backbone(self, backbone, x):
        """Forward pass through the backbone"""
        b, c, h, w = x.shape
        
        # Use regular backbone forward method
        activation_dict = {}
        
        def hook_fn(module, input, output, name):
            activation_dict[name] = output
            
        handles = []
        for i, block in enumerate(backbone.blocks):
            if i in [5, 11, 17, 23]:  # Assuming these are the hook points
                handle = block.register_forward_hook(
                    lambda module, input, output, name=i: hook_fn(module, input, output, name)
                )
                handles.append(handle)
                
        _ = backbone(x)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
            
        # Get activations
        layer_1 = activation_dict.get(5, None)
        layer_2 = activation_dict.get(11, None)
        layer_3 = activation_dict.get(17, None)
        layer_4 = activation_dict.get(23, None)
        
        return layer_1, layer_2, layer_3, layer_4


class LSegEncBLIPNet(LSegEncBLIP):
    """Network for semantic segmentation with BLIP."""
    def __init__(self, labels, path=None, scale_factor=0.5, crop_size=480, **kwargs):
        features = kwargs["features"] if "features" in kwargs else 256
        kwargs["use_bn"] = True

        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.labels = labels

        head = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)
