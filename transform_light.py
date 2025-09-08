# 轻量化数据增强策略
import random
from torchvision import transforms
import torchvision.transforms.functional as F

class PairedLightTrainTransform:
    """
    轻量化训练数据增强 - 减少过度增强，提高训练稳定性
    只使用3种增强：原图、水平翻转、垂直翻转
    """
    def __init__(self, resize_shorter=512, output_size=(512, 512)):
        self.resize_shorter = resize_shorter
        self.output_size = output_size

    def resize_shorter_edge(self, img):
        w, h = img.size
        if w < h:
            new_w = self.resize_shorter
            new_h = int(h * self.resize_shorter / w)
        else:
            new_h = self.resize_shorter
            new_w = int(w * self.resize_shorter / h)
        return F.resize(img, (new_h, new_w), interpolation=transforms.InterpolationMode.BILINEAR)

    def __call__(self, rgb_img, uv_img):
        # 预处理：调整尺寸并中心裁剪
        rgb_img = self.resize_shorter_edge(rgb_img)
        uv_img = self.resize_shorter_edge(uv_img)
        
        rgb_img = F.center_crop(rgb_img, self.output_size)
        uv_img = F.center_crop(uv_img, self.output_size)

        imgs_rgb, imgs_uv = [], []

        # 1. 原图
        imgs_rgb.append(F.to_tensor(rgb_img))
        imgs_uv.append(F.to_tensor(uv_img))

        # 2. 水平翻转
        imgs_rgb.append(F.to_tensor(F.hflip(rgb_img)))
        imgs_uv.append(F.to_tensor(F.hflip(uv_img)))

        # 3. 垂直翻转
        imgs_rgb.append(F.to_tensor(F.vflip(rgb_img)))
        imgs_uv.append(F.to_tensor(F.vflip(uv_img)))

        return imgs_rgb, imgs_uv

class PairedMinimalTrainTransform:
    """
    最小化训练数据增强 - 仅使用水平翻转
    适用于极小数据集或调试阶段
    """
    def __init__(self, resize_shorter=512, output_size=(512, 512)):
        self.resize_shorter = resize_shorter
        self.output_size = output_size

    def resize_shorter_edge(self, img):
        w, h = img.size
        if w < h:
            new_w = self.resize_shorter
            new_h = int(h * self.resize_shorter / w)
        else:
            new_h = self.resize_shorter
            new_w = int(w * self.resize_shorter / h)
        return F.resize(img, (new_h, new_w), interpolation=transforms.InterpolationMode.BILINEAR)

    def __call__(self, rgb_img, uv_img):
        rgb_img = self.resize_shorter_edge(rgb_img)
        uv_img = self.resize_shorter_edge(uv_img)
        
        rgb_img = F.center_crop(rgb_img, self.output_size)
        uv_img = F.center_crop(uv_img, self.output_size)

        imgs_rgb, imgs_uv = [], []

        # 1. 原图
        imgs_rgb.append(F.to_tensor(rgb_img))
        imgs_uv.append(F.to_tensor(uv_img))

        # 2. 水平翻转
        imgs_rgb.append(F.to_tensor(F.hflip(rgb_img)))
        imgs_uv.append(F.to_tensor(F.hflip(uv_img)))

        return imgs_rgb, imgs_uv

class PairedStableTrainTransform:
    """
    稳定训练增强策略 - 包含轻微的随机变换
    在基础翻转基础上添加小幅度旋转和亮度调整
    """
    def __init__(self, resize_shorter=512, output_size=(512, 512)):
        self.resize_shorter = resize_shorter
        self.output_size = output_size
        self.rotation_degree = 10  # 减小旋转角度

    def resize_shorter_edge(self, img):
        w, h = img.size
        if w < h:
            new_w = self.resize_shorter
            new_h = int(h * self.resize_shorter / w)
        else:
            new_h = self.resize_shorter
            new_w = int(w * self.resize_shorter / h)
        return F.resize(img, (new_h, new_w), interpolation=transforms.InterpolationMode.BILINEAR)

    def paired_random_rotation(self, img1, img2, degree):
        angle = random.uniform(-degree, degree)
        img1 = F.rotate(img1, angle)
        img2 = F.rotate(img2, angle)
        return img1, img2

    def __call__(self, rgb_img, uv_img):
        rgb_img = self.resize_shorter_edge(rgb_img)
        uv_img = self.resize_shorter_edge(uv_img)
        
        rgb_img = F.center_crop(rgb_img, self.output_size)
        uv_img = F.center_crop(uv_img, self.output_size)

        imgs_rgb, imgs_uv = [], []

        # 1. 原图
        imgs_rgb.append(F.to_tensor(rgb_img))
        imgs_uv.append(F.to_tensor(uv_img))

        # 2. 水平翻转
        imgs_rgb.append(F.to_tensor(F.hflip(rgb_img)))
        imgs_uv.append(F.to_tensor(F.hflip(uv_img)))

        # 3. 垂直翻转
        imgs_rgb.append(F.to_tensor(F.vflip(rgb_img)))
        imgs_uv.append(F.to_tensor(F.vflip(uv_img)))

        # 4. 小幅度随机旋转
        rot_rgb, rot_uv = self.paired_random_rotation(rgb_img, uv_img, self.rotation_degree)
        imgs_rgb.append(F.to_tensor(rot_rgb))
        imgs_uv.append(F.to_tensor(rot_uv))

        return imgs_rgb, imgs_uv

class PairedTTAEvalTransform:
    """
    测试时增强(TTA)验证变换
    返回多个版本用于集成预测
    """
    def __init__(self, resize_shorter=512, output_size=(512, 512)):
        self.resize_shorter = resize_shorter
        self.output_size = output_size

    def resize_shorter_edge(self, img):
        w, h = img.size
        if w < h:
            new_w = self.resize_shorter
            new_h = int(h * self.resize_shorter / w)
        else:
            new_h = self.resize_shorter
            new_w = int(w * self.resize_shorter / h)
        return F.resize(img, (new_h, new_w), interpolation=transforms.InterpolationMode.BILINEAR)

    def __call__(self, rgb_img, uv_img):
        rgb_img = self.resize_shorter_edge(rgb_img)
        uv_img = self.resize_shorter_edge(uv_img)
        
        rgb_img = F.center_crop(rgb_img, self.output_size)
        uv_img = F.center_crop(uv_img, self.output_size)

        imgs_rgb, imgs_uv = [], []

        # 原图
        imgs_rgb.append(F.to_tensor(rgb_img))
        imgs_uv.append(F.to_tensor(uv_img))

        # 水平翻转
        imgs_rgb.append(F.to_tensor(F.hflip(rgb_img)))
        imgs_uv.append(F.to_tensor(F.hflip(uv_img)))

        return imgs_rgb, imgs_uv

# 使用建议：
# 1. 如果过拟合严重，使用 PairedMinimalTrainTransform (2倍增强)
# 2. 如果需要平衡，使用 PairedLightTrainTransform (3倍增强)  
# 3. 如果数据质量好，使用 PairedStableTrainTransform (4倍增强)
# 4. 验证时可以使用 PairedTTAEvalTransform 进行测试时增强
