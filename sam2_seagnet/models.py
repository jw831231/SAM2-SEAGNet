import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


class CBAM(nn.Module):
    """Convolutional Block Attention Module (user's version)"""
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_att

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.conv(concat))
        x = x * spatial_att
        return x


class SpeckleAndEdgeAwareAttentionGate(nn.Module):
    """Speckle- and Edge-Aware Attention Gate (SEAG)"""
    def __init__(self, current_channels: int, skip_channels: int):
        super().__init__()
        gate_in_channels = current_channels + skip_channels + 2
        self.skip_proj = nn.Conv2d(skip_channels, current_channels, kernel_size=1, bias=False)
        self.gate_conv = nn.Conv2d(gate_in_channels, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, current_feat, skip_feat):
        # Speckle map (local variance)
        skip_gray = skip_feat.mean(dim=1, keepdim=True)
        skip_mean = F.avg_pool2d(skip_gray, kernel_size=3, stride=1, padding=1)
        skip_sq_mean = F.avg_pool2d(skip_gray ** 2, kernel_size=3, stride=1, padding=1)
        speckle_map = torch.sqrt(torch.clamp(skip_sq_mean - skip_mean ** 2, min=1e-6))

        # Edge map (Sobel)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                               dtype=skip_feat.dtype, device=skip_feat.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                               dtype=skip_feat.dtype, device=skip_feat.device).view(1, 1, 3, 3)

        grad_x = F.conv2d(skip_gray, sobel_x, padding=1)
        grad_y = F.conv2d(skip_gray, sobel_y, padding=1)
        edge_map = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        # Attention
        concat = torch.cat([current_feat, skip_feat, speckle_map, edge_map], dim=1)
        attention = self.sigmoid(self.gate_conv(concat))

        weighted_skip = skip_feat * attention
        weighted_skip = self.skip_proj(weighted_skip)
        out = current_feat + weighted_skip
        return out


class ASPP(nn.Module):
    """Customized Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels=512, atrous_rates=[6, 12], branch_channels=128, out_channels=512):
        super().__init__()
        self.branches = nn.ModuleList()

        # 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        ))

        # Atrous branches
        for rate in atrous_rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=rate, dilation=rate,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True)
            ))

        # Global average pooling branch
        self.branches.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        ))

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(len(self.branches) * branch_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch_outputs = []
        for i, branch in enumerate(self.branches):
            out = branch(x)
            if i == len(self.branches) - 1:  # global pool branch
                out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
            branch_outputs.append(out)

        fused = torch.cat(branch_outputs, dim=1)
        fused = self.fusion_conv(fused)
        return fused


class SAM2FeatureExtractor(nn.Module):
    """Extract features from frozen SAM2 image encoder"""
    def __init__(self, sam_model):
        super().__init__()
        self.image_encoder = sam_model.image_encoder
        self.trunk = sam_model.image_encoder.trunk

    def forward(self, x):
        full_dict = self.image_encoder(x)
        raw_stages = self.trunk(x)
        stage1 = raw_stages[0]
        stage2 = raw_stages[1] if len(raw_stages) > 1 else None
        return {
            'vision_features': full_dict['vision_features'],
            'stage1': stage1,
            'stage2': stage2
        }


class SAM2PromptGenerator(nn.Module):
    """Main SAM2-SEAGNet model"""
    def __init__(self, sam_model, num_classes=1):
        super().__init__()
        self.feature_extractor = SAM2FeatureExtractor(sam_model)

        # Vision features projection
        self.vision_proj = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )

        self.aspp = ASPP(in_channels=512)

        # SEAG modules
        self.gate_layer1 = SpeckleAndEdgeAwareAttentionGate(current_channels=256, skip_channels=224)
        self.gate_layer2 = SpeckleAndEdgeAwareAttentionGate(current_channels=128, skip_channels=112)

        # CBAM modules
        self.cbam1 = CBAM(channels=256)
        self.cbam2 = CBAM(channels=128)
        self.cbam3 = CBAM(channels=64)
        self.cbam4 = CBAM(channels=32)

        # 4-layer decoder
        self.decoder_layer1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )
        self.decoder_layer2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )
        self.decoder_layer3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )
        self.decoder_layer4 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        with autocast(device_type='cuda'):
            feats_dict = self.feature_extractor(x)

            vision = feats_dict['vision_features']
            vision = self.vision_proj(vision)
            bottleneck = self.aspp(vision)

            # Layer 1
            x = F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=False)
            x = self.decoder_layer1(x)
            x = self.gate_layer1(x, feats_dict['stage2'])
            x = self.cbam1(x)

            # Layer 2
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = self.decoder_layer2(x)
            x = self.gate_layer2(x, feats_dict['stage1'])
            x = self.cbam2(x)

            # Layer 3 (no skip)
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = self.decoder_layer3(x)
            x = self.cbam3(x)

            # Layer 4 (no skip)
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = self.decoder_layer4(x)
            x = self.cbam4(x)

            out = self.final_conv(x)
            return out
