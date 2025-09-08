import torch
import torch.nn as nn
import timm
from safetensors.torch import load_file

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale

class DualBranchLightweight(nn.Module):
    """
    轻量化双分支模型 - 使用ResNet18作为backbone
    适合小数据集训练，参数量约22M（相比InceptionV3的44M）
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        # 使用ResNet18作为特征提取器
        self.rgb_branch = timm.create_model("resnet18", pretrained=pretrained, num_classes=0, global_pool="")
        self.uv_branch = timm.create_model("resnet18", pretrained=pretrained, num_classes=0, global_pool="")
        
        # ResNet18最后一层特征维度是512
        feature_dim = 512
        
        # SE注意力机制
        self.se_rgb = SEBlock(channels=feature_dim)
        self.se_uv = SEBlock(channels=feature_dim)
        
        # 融合后的卷积层
        self.post_fuse_conv = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # 输出4x4特征图
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear((feature_dim // 2) * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, rgb, uv):
        # 特征提取
        rgb_feats = self.rgb_branch(rgb)  # [B, 512, H, W]
        uv_feats = self.uv_branch(uv)    # [B, 512, H, W]
        
        # 注意力机制
        rgb_feats = self.se_rgb(rgb_feats)
        uv_feats = self.se_uv(uv_feats)
        
        # 特征融合
        fused = torch.cat([rgb_feats, uv_feats], dim=1)  # [B, 1024, H, W]
        
        # 后处理
        fused = self.post_fuse_conv(fused)  # [B, 256, 4, 4]
        
        # 分类
        logits = self.classifier(fused)
        
        return logits

class DualBranchMini(nn.Module):
    """
    超轻量化双分支模型 - 使用MobileNetV3作为backbone
    适合极小数据集或快速原型验证，参数量约10M
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        # 使用MobileNetV3作为特征提取器
        self.rgb_branch = timm.create_model("mobilenetv3_small_100", pretrained=pretrained, num_classes=0, global_pool="")
        self.uv_branch = timm.create_model("mobilenetv3_small_100", pretrained=pretrained, num_classes=0, global_pool="")
        
        # MobileNetV3-Small最后一层特征维度
        feature_dim = 576
        
        # 简化的融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 简化的分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, rgb, uv):
        rgb_feats = self.rgb_branch(rgb)
        uv_feats = self.uv_branch(uv)
        
        fused = torch.cat([rgb_feats, uv_feats], dim=1)
        fused = self.fusion(fused)
        logits = self.classifier(fused)
        
        return logits

class DualBranchEfficientNet(nn.Module):
    """
    中等复杂度双分支模型 - 使用EfficientNet-B0
    平衡性能和效率，参数量约10M
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        # 使用EfficientNet-B0
        self.rgb_branch = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0, global_pool="")
        self.uv_branch = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0, global_pool="")
        
        # EfficientNet-B0最后一层特征维度
        feature_dim = 1280
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.Swish(),
            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.Swish(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(feature_dim // 2, 128),
            nn.Swish(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, rgb, uv):
        rgb_feats = self.rgb_branch(rgb)
        uv_feats = self.uv_branch(uv)
        
        fused = torch.cat([rgb_feats, uv_feats], dim=1)
        fused = self.fusion(fused)
        logits = self.classifier(fused)
        
        return logits

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 测试不同模型的参数量和输出
    models = {
        "DualBranchLightweight": DualBranchLightweight(),
        "DualBranchMini": DualBranchMini(),
        "DualBranchEfficientNet": DualBranchEfficientNet()
    }
    
    dummy_rgb = torch.randn(1, 3, 512, 512)
    dummy_uv = torch.randn(1, 3, 512, 512)
    
    for name, model in models.items():
        try:
            outputs = model(dummy_rgb, dummy_uv)
            params = count_parameters(model)
            print(f"{name}:")
            print(f"  参数量: {params:,} ({params/1e6:.1f}M)")
            print(f"  输出形状: {outputs.shape}")
            print()
        except Exception as e:
            print(f"{name} 测试失败: {e}")

# 使用建议：
# 1. 数据量 < 1000: 使用 DualBranchMini
# 2. 数据量 1000-5000: 使用 DualBranchLightweight  
# 3. 数据量 > 5000: 使用 DualBranchEfficientNet
# 4. 如果GPU内存不足，优先选择 DualBranchMini
