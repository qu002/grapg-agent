import torch
import torch.nn as nn
import timm
from safetensors.torch import load_file
class SEBlock(nn.Module):
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

class DualBranchInceptionV3(nn.Module):
    def __init__(self, num_classes=2, weight_path=None):
        super().__init__()
        def create_branch():
            base = timm.create_model("inception_v3", pretrained=False, num_classes=0, global_pool="")
            return nn.Sequential(base)
        self.rgb_branch = create_branch()
        self.uv_branch = create_branch()
        if weight_path is not None:
            state_dict = load_file(weight_path)
            self.rgb_branch[0].load_state_dict(state_dict, strict=False)
            self.uv_branch[0].load_state_dict(state_dict, strict=False)
        # self.se_rgb = SEBlock(channels=2048)
        # self.se_uv = SEBlock(channels=2048)
        self.se_fused = SEBlock(channels=4096)
        # 融合后多层卷积和池化，将空间尺寸调整到7x7
        self.post_fuse_conv = nn.Sequential(
            nn.Conv2d(4096, 1024, kernel_size=3, padding=1),  # 保持尺寸
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),   # 保持尺寸
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))  # 将空间维度调整为7x7
        )
        # 分类器，输入是 512 * 7 * 7
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, rgb, uv):
        rgb_feats = self.rgb_branch(rgb)   # [B, 2048, 32, W]
        # print(rgb_feats.shape)
        uv_feats = self.uv_branch(uv)      # [B, 2048, 32, W]
        # print(uv_feats.shape)
        # rgb_feats = self.se_rgb(rgb_feats)
        # # print(rgb_feats.shape)
        # uv_feats = self.se_uv(uv_feats)
        # print(uv_feats.shape)
        fused = torch.cat([rgb_feats, uv_feats], dim=1)  # [B, 4096, H, W]
        fused = self.se_fused(fused)
        # print(fused.shape)
        fused = self.post_fuse_conv(fused)  # [B, 512, 7, 7]
        # print(fused.shape)
        logits = self.classifier(fused)     # [B, num_classes]
        # print(logits.shape)
        return logits


if __name__ == "__main__":
    model = DualBranchInceptionV3()
    dummy_rgb = torch.randn(1, 3, 1088,1088)
    dummy_uv = torch.randn(1, 3, 1088, 1088)
    outputs = model(dummy_rgb, dummy_uv)
    # print(outputs.shape)
    #right
# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super().__init__()
#         self.se = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, channels // reduction, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // reduction, channels, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         scale = self.se(x)
#         return x * scale
#
# class DualBranchInceptionV3(nn.Module):
#     def __init__(self, num_classes=2, weight_path=None):
#         super().__init__()
#         def create_branch():
#             base = timm.create_model("inception_v3", pretrained=False, num_classes=0, global_pool="")
#             return nn.Sequential(base)
#         self.rgb_branch = create_branch()
#         self.uv_branch = create_branch()
#         if weight_path is not None:
#             state_dict = load_file(weight_path)
#             self.rgb_branch[0].load_state_dict(state_dict, strict=False)
#             self.uv_branch[0].load_state_dict(state_dict, strict=False)
#         self.se_rgb = SEBlock(channels=2048)
#         self.se_uv = SEBlock(channels=2048)
#         # 融合后多层卷积和池化，将空间尺寸调整到7x7
#         self.post_fuse_conv = nn.Sequential(
#             nn.Conv2d(4096, 1024, kernel_size=3, padding=1),  # 保持尺寸
#             nn.BatchNorm2d(1024),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(1024, 512, kernel_size=3, padding=1),   # 保持尺寸
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d((7, 7))  # 将空间维度调整为7x7
#         )
#         # 分类器，输入是 512 * 7 * 7
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(512 * 7 * 7, 256),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, num_classes)
#         )
#     def forward(self, rgb, uv):
#         rgb_feats = self.rgb_branch(rgb)   # [B, 2048, 32, W]
#         # print(rgb_feats.shape)
#         uv_feats = self.uv_branch(uv)      # [B, 2048, 32, W]
#         # print(uv_feats.shape)
#         rgb_feats = self.se_rgb(rgb_feats)
#         # print(rgb_feats.shape)
#         uv_feats = self.se_uv(uv_feats)
#         # print(uv_feats.shape)
#         fused = torch.cat([rgb_feats, uv_feats], dim=1)  # [B, 4096, H, W]
#         # print(fused.shape)
#         fused = self.post_fuse_conv(fused)  # [B, 512, 7, 7]
#         # print(fused.shape)
#         logits = self.classifier(fused)     # [B, num_classes]
#         # print(logits.shape)
#         return logits
#
#
# if __name__ == "__main__":
#     model = DualBranchInceptionV3()
#     dummy_rgb = torch.randn(1, 3, 1088,1088)
#     dummy_uv = torch.randn(1, 3, 1088, 1088)
#     outputs = model(dummy_rgb, dummy_uv)
#     # print(outputs.shape)

# import torch
# import torch.nn as nn
# import timm
# from safetensors.torch import load_file
#
# class DualBranchResNet(nn.Module):
#     def __init__(self, backbone_name="resnet18.a1_in1k", num_classes=2, weight_path=None):
#         super().__init__()
#
#         # RGB 分支，不加载预训练权重，后面手动加载
#         self.rgb_branch = timm.create_model(
#             backbone_name,
#             pretrained=False,
#             features_only=True
#         )
#         # UV 分支，同上
#         self.uv_branch = timm.create_model(
#             backbone_name,
#             pretrained=False,
#             features_only=True
#         )
#
#         # 如果给了权重路径，手动加载权重
#         if weight_path is not None:
#             # 读取safetensors权重字典
#             state_dict = load_file(weight_path)
#             # timm模型权重key可能带有前缀，需调整或者直接加载
#             self.rgb_branch.load_state_dict(state_dict, strict=False)
#             self.uv_branch.load_state_dict(state_dict, strict=False)
#
#         # 获取最后一个输出层的通道数
#         out_channels = self.rgb_branch.feature_info.channels()[-1]
#
#         self.classifier = nn.Sequential(
#             nn.Linear(out_channels * 2, 256),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, num_classes)
#         )
#
#     def forward(self, rgb, uv):
#         rgb_feats = self.rgb_branch(rgb)[-1]
#         uv_feats = self.uv_branch(uv)[-1]
#
#         rgb_pool = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(rgb_feats, 1), 1)
#         uv_pool = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(uv_feats, 1), 1)
#
#         fused = torch.cat([rgb_pool, uv_pool], dim=1)
#         logits = self.classifier(fused)
#         return logits
# if __name__ == "__main__":
#     model = DualBranchResNet()
#     dummy_rgb = torch.randn(1, 3, 1024,1024)
#     dummy_uv = torch.randn(1, 3, 1024, 1024)
#     outputs = model(dummy_rgb, dummy_uv)
#     print(outputs.shape)
# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super().__init__()
#         self.se = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, channels // reduction, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // reduction, channels, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         scale = self.se(x)
#         return x * scale
#
# class DualBranchResNet(nn.Module):
#     def __init__(self,  num_classes=2, weight_path=None):
#         super().__init__()
#
#         def create_branch():
#             base = timm.create_model("resnet18.a1_in1k", pretrained=False, num_classes=0, global_pool="")
#             return nn.Sequential(base)
#
#         self.rgb_branch = create_branch()
#         self.uv_branch = create_branch()
#
#         # 如果给了权重路径，手动加载权重
#         if weight_path is not None:
#             # 读取safetensors权重字典
#             state_dict = load_file(weight_path)
#             # timm模型权重key可能带有前缀，需调整或者直接加载
#             self.rgb_branch[0].load_state_dict(state_dict, strict=False)
#             self.uv_branch[0].load_state_dict(state_dict, strict=False)
#
#
#
#         self.se_rgb=SEBlock(channels=512)
#         self.se_uv=SEBlock(channels=512)
#         self.post_fuse_conv = nn.Sequential(
#             nn.Conv2d(1024, 512, kernel_size=3, padding=1),  # 保持尺寸
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(512, 256, kernel_size=3, padding=1),   # 保持尺寸
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#
#             nn.AdaptiveAvgPool2d((7, 7))
#         ) # 将空间维度调整为7x7
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(256* 7 * 7, 256),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, num_classes)
#         )
#     def forward(self, rgb, uv):
#         rgb_feats = self.rgb_branch(rgb)   # [B, 2048, 32, W]
#         # print(rgb_feats.shape)
#         uv_feats = self.uv_branch(uv)      # [B, 2048, 32, W]
#         # print(uv_feats.shape)
#         rgb_feats = self.se_rgb(rgb_feats)
#         # print(rgb_feats.shape)
#         uv_feats = self.se_uv(uv_feats)
#         # print(uv_feats.shape)
#         fused = torch.cat([rgb_feats, uv_feats], dim=1)  # [B, 4096, H, W]
#         # print(fused.shape)
#         fused = self.post_fuse_conv(fused)  # [B, 512, 7, 7]
#         # print(fused.shape)
#         logits = self.classifier(fused)     # [B, num_classes]
#         # print(logits.shape)
#         return logits
# if __name__ == "__main__":
#     model = DualBranchResNet()
#     dummy_rgb = torch.randn(1, 3, 1024,1024)
#     dummy_uv = torch.randn(1, 3, 1024, 1024)
#     outputs = model(dummy_rgb, dummy_uv)
#     # print(outputs.shape)

