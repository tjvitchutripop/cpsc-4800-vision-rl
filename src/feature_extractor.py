import torch
import torch.nn as nn
import transformers
from transformers import AutoModel
from torchvision import models
from torchvision.models import ResNet50_Weights, DenseNet121_Weights, EfficientNet_B0_Weights

class NatureCNN(nn.Module):
    def __init__(self, sample_obs):
        super().__init__()

        extractors = {}

        self.out_features = 0
        feature_size = 256
        in_channels = 0
        if "rgb" in sample_obs:
            in_channels += sample_obs["rgb"].shape[-1]
            image_size = (sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])
        if "depth" in sample_obs:
            in_channels += sample_obs["depth"].shape[-1]
            image_size = (sample_obs["depth"].shape[1], sample_obs["depth"].shape[2])
        if "segmentation" in sample_obs:
            in_channels += sample_obs["segmentation"].shape[-1]
            image_size = (sample_obs["segmentation"].shape[1], sample_obs["segmentation"].shape[2])


        # here we use a NatureCNN architecture to process images, but any architecture is permissble here
        cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        # to easily figure out the dimensions after flattening, we pass a test tensor
        with torch.no_grad():
            # Build a sample image tensor combining available modalities (rgb/depth/segmentation)
            image_tensors = []
            if "rgb" in sample_obs:
                rgb = sample_obs["rgb"].float().permute(0, 3, 1, 2).cpu() / 255.0
                image_tensors.append(rgb)
            if "depth" in sample_obs:
                depth = sample_obs["depth"].float().permute(0, 3, 1, 2).cpu()
                # If depth is not normalized, keep as float; CNN will learn scale
                image_tensors.append(depth)
            if "segmentation" in sample_obs:
                seg = sample_obs["segmentation"].float().permute(0, 3, 1, 2).cpu()
                image_tensors.append(seg)
            assert len(image_tensors) > 0, "No image modalities found in observations"
            sample_image = torch.cat(image_tensors, dim=1)
            n_flatten = cnn(sample_image).shape[1]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        # Single image extractor that handles combined rgb+depth channels
        extractors["image"] = nn.Sequential(cnn, fc)
        self.out_features += feature_size

        if "state" in sample_obs:
            # for state data we simply pass it through a single linear layer
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key == "image":
                image_tensors = []
                if "rgb" in observations:
                    rgb = observations["rgb"].float().permute(0, 3, 1, 2) / 255.0
                    image_tensors.append(rgb)
                if "depth" in observations:
                    depth = observations["depth"].float().permute(0, 3, 1, 2)
                    image_tensors.append(depth)
                if "segmentation" in observations:
                    seg = observations["segmentation"].float().permute(0, 3, 1, 2)
                    image_tensors.append(seg)
                obs_img = torch.cat(image_tensors, dim=1)
                encoded_tensor_list.append(extractor(obs_img))
            else:
                obs = observations[key]
                encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)

# From https://theia.theaiinstitute.com
class Theia(nn.Module):
    def __init__(self, sample_obs=None, model_name="theaiinstitute/theia-tiny-patch16-224-cdiv", freeze_backbone=True):
        super().__init__()
        
        # Load pretrained Theia model
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        print(f"Loaded Theia model: {model_name}")
        
        # Freeze backbone weights if specified (recommended for faster training)
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Expected input size for Theia (typically 224x224)
        self.expected_size = 224
        
        # Theia outputs sequence format: (B, num_patches, hidden_dim)
        # For theia-base with 224x224 input and patch size 16: (B, 196, 768)
        # 196 = 14×14 patches, 768 = hidden dimension
        # We need to reshape to spatial format then process with conv neck
        
        self.patch_grid_size = 14  # sqrt(196) = 14
        self.hidden_dim = 192  # theia-tiny hidden dimension
        
        # Neck: Convolutional layers to process spatial features
        # Similar to ConvPolicyHead from the Theia repo
        self.neck = nn.Sequential(
            nn.Conv2d(self.hidden_dim, 256, kernel_size=4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.LayerNorm([256, 7, 7]),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2),  # 7x7 -> 3x3
            nn.LayerNorm([256, 3, 3]),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),  # 3x3 -> 1x1
            nn.LayerNorm([256, 1, 1]),
            nn.ReLU(),
            nn.Flatten(),  # -> (B, 256)
        )
        
        self.out_features = 256
    
    def forward(self, observations) -> torch.Tensor:
        # Extract RGB images from observations
        # Theia expects RGB only, not depth or segmentation
        if "rgb" in observations:
            # observations["rgb"] shape: (batch, height, width, channels)
            rgb = observations["rgb"].float() / 255.0  # Normalize to [0, 1]
            
            # Permute to (batch, channels, height, width)
            rgb = rgb.permute(0, 3, 1, 2)
            
            # Resize if necessary (Theia typically expects 224x224)
            if rgb.shape[-2:] != (self.expected_size, self.expected_size):
                rgb = torch.nn.functional.interpolate(
                    rgb, 
                    size=(self.expected_size, self.expected_size), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Extract spatial features using Theia
            # forward_feature returns (B, num_patches, hidden_dim) sequence format
            with torch.no_grad(): #if not self.model.training else torch.enable_grad():
                theia_patch_features = self.model.forward_feature(rgb)
                # Expected shape: (B, 196, 192) for theia-tiny-patch16-224
                
                # Reshape from sequence format (B, N, D) to spatial format (B, D, H, W)
                # 196 patches = 14×14 grid
                batch_size = theia_patch_features.shape[0]
                theia_spatial_features = theia_patch_features.transpose(1, 2).reshape(
                    batch_size, self.hidden_dim, self.patch_grid_size, self.patch_grid_size
                )
                # Now shape: (B, 768, 14, 14)
            
            # Process spatial features through the neck
            # The neck is always trainable even if backbone is frozen
            features = self.neck(theia_spatial_features)

            return features
        else:
            raise ValueError("Theia feature extractor requires 'rgb' observations")


class ResNet50(nn.Module):
    def __init__(self, sample_obs=None, freeze_backbone=True):
        super().__init__()
        
        # Load pretrained ResNet50 with ImageNet weights
        weights = ResNet50_Weights.IMAGENET1K_V2  # Using the latest v2 weights
        resnet = models.resnet50(weights=weights)
        print(f"Loaded ResNet50 with {weights} pretrained weights")
        
        # Remove the final classification layer (fc)
        # ResNet50 outputs 2048 features from avgpool before the fc layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove fc layer
        
        # Freeze backbone weights if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Expected input size for ResNet (typically 224x224)
        self.expected_size = 224
        
        # ImageNet normalization (required for pretrained models)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # ResNet50 outputs 2048 features, project to 256 to match other extractors
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        self.out_features = 256
    
    def forward(self, observations) -> torch.Tensor:
        # Extract RGB images from observations
        if "rgb" in observations:
            # observations["rgb"] shape: (batch, height, width, channels)
            rgb = observations["rgb"].float() / 255.0  # Normalize to [0, 1]
            
            # Permute to (batch, channels, height, width)
            rgb = rgb.permute(0, 3, 1, 2)
            
            # Resize if necessary (ResNet expects 224x224)
            if rgb.shape[-2:] != (self.expected_size, self.expected_size):
                rgb = torch.nn.functional.interpolate(
                    rgb, 
                    size=(self.expected_size, self.expected_size), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Apply ImageNet normalization (critical for pretrained models!)
            rgb = (rgb - self.mean) / self.std
            
            # Extract features using ResNet50 backbone
            with torch.no_grad() if not self.backbone.training else torch.enable_grad():
                resnet_features = self.backbone(rgb)
                # Output shape: (batch, 2048, 1, 1) after avgpool
            
            # Project to desired feature dimension
            # The projection head is always trainable
            features = self.projection(resnet_features)
            
            return features
        else:
            raise ValueError("ResNet50 feature extractor requires 'rgb' observations")

class EfficientNetB0(nn.Module):
    def __init__(self, sample_obs=None, freeze_backbone=True):
        super().__init__()
        
        # Load pretrained EfficientNet-B0 with ImageNet weights
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        efficientnet = models.efficientnet_b0(weights=weights)
        print(f"Loaded EfficientNet-B0 with {weights} pretrained weights")
        
        # EfficientNet has features + classifier structure
        # Remove the final classification layer
        # EfficientNet-B0 outputs 1280 features from the final layer
        self.backbone = efficientnet.features
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        # Freeze backbone weights if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Expected input size for EfficientNet-B0 (224x224)
        self.expected_size = 224
        
        # ImageNet normalization (required for pretrained models)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # EfficientNet-B0 outputs 1280 features, project to 256 to match other extractors
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        self.out_features = 256
    
    def forward(self, observations) -> torch.Tensor:
        # Extract RGB images from observations
        if "rgb" in observations:
            # observations["rgb"] shape: (batch, height, width, channels)
            rgb = observations["rgb"].float() / 255.0  # Normalize to [0, 1]
            
            # Permute to (batch, channels, height, width)
            rgb = rgb.permute(0, 3, 1, 2)
            
            # Resize if necessary (EfficientNet-B0 expects 224x224)
            if rgb.shape[-2:] != (self.expected_size, self.expected_size):
                rgb = torch.nn.functional.interpolate(
                    rgb, 
                    size=(self.expected_size, self.expected_size), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Apply ImageNet normalization (critical for pretrained models!)
            rgb = (rgb - self.mean) / self.std
            
            # Extract features using EfficientNet-B0 backbone
            with torch.no_grad() if not self.backbone.training else torch.enable_grad():
                efficientnet_features = self.backbone(rgb)
                # Output shape: (batch, 1280, H, W) where H,W depend on input size
                efficientnet_features = self.pooling(efficientnet_features)
                # Output shape: (batch, 1280, 1, 1)
            
            # Project to desired feature dimension
            # The projection head is always trainable
            features = self.projection(efficientnet_features)
            
            return features
        else:
            raise ValueError("EfficientNet-B0 feature extractor requires 'rgb' observations")

class DenseNet121(nn.Module):
    def __init__(self, sample_obs=None, freeze_backbone=True):
        super().__init__()
        
        # Load pretrained DenseNet121 with ImageNet weights
        weights = DenseNet121_Weights.IMAGENET1K_V1
        densenet = models.densenet121(weights=weights)
        print(f"Loaded DenseNet121 with {weights} pretrained weights")
        
        # DenseNet has features + classifier structure
        # Remove the final classification layer
        # DenseNet121 outputs 1024 features from the final layer
        self.backbone = densenet.features
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        # Freeze backbone weights if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Expected input size for DenseNet (typically 224x224)
        self.expected_size = 224
        
        # ImageNet normalization (required for pretrained models)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # DenseNet121 outputs 1024 features, project to 256 to match other extractors
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        self.out_features = 256
    
    def forward(self, observations) -> torch.Tensor:
        # Extract RGB images from observations
        if "rgb" in observations:
            # observations["rgb"] shape: (batch, height, width, channels)
            rgb = observations["rgb"].float() / 255.0  # Normalize to [0, 1]
            
            # Permute to (batch, channels, height, width)
            rgb = rgb.permute(0, 3, 1, 2)
            
            # Resize if necessary (DenseNet expects 224x224)
            if rgb.shape[-2:] != (self.expected_size, self.expected_size):
                rgb = torch.nn.functional.interpolate(
                    rgb, 
                    size=(self.expected_size, self.expected_size), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Apply ImageNet normalization (critical for pretrained models!)
            rgb = (rgb - self.mean) / self.std
            
            # Extract features using DenseNet121 backbone
            with torch.no_grad() if not self.backbone.training else torch.enable_grad():
                densenet_features = self.backbone(rgb)
                # Output shape: (batch, 1024, H, W) where H,W depend on input size
                densenet_features = self.pooling(densenet_features)
                # Output shape: (batch, 1024, 1, 1)
            
            # Project to desired feature dimension
            # The projection head is always trainable
            features = self.projection(densenet_features)
            
            return features
        else:
            raise ValueError("DenseNet121 feature extractor requires 'rgb' observations")