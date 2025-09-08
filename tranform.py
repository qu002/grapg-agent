# dataset/transforms.py
import random
from torchvision import transforms
import torchvision.transforms.functional as F
# 用于训练集，返回多种增强版本
class PairedTrainTransform:
    def __init__(self, resize_shorter=1088, output_size=(1088, 1088)):
        self.resize_shorter = resize_shorter
        self.output_size = output_size
        self.scale_range = (0.8, 1.0)  # 随机缩放比例
        self.rotation_degree = 20      # 随机旋转最大角度

    def resize_shorter_edge(self, img):
        w, h = img.size
        if w < h:
            new_w = self.resize_shorter
            new_h = int(h * self.resize_shorter / w)
        else:
            new_h = self.resize_shorter
            new_w = int(w * self.resize_shorter / h)
        return F.resize(img, (new_h, new_w), interpolation=transforms.InterpolationMode.BILINEAR)

    def paired_random_resized_crop(self, img1, img2):
        # 生成相同参数的随机裁剪区域
        scale = random.uniform(*self.scale_range)
        i, j, h, w = transforms.RandomResizedCrop.get_params(img1, scale=(scale, 1.0), ratio=(1.0, 1.0))
        img1 = F.resized_crop(img1, i, j, h, w, self.output_size)
        img2 = F.resized_crop(img2, i, j, h, w, self.output_size)
        return img1, img2

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

        # 原图
        imgs_rgb.append(F.to_tensor(rgb_img))
        imgs_uv.append(F.to_tensor(uv_img))

        # vflip
        imgs_rgb.append(F.to_tensor(F.vflip(rgb_img)))
        imgs_uv.append(F.to_tensor(F.vflip(uv_img)))

        # hflip
        imgs_rgb.append(F.to_tensor(F.hflip(rgb_img)))
        imgs_uv.append(F.to_tensor(F.hflip(uv_img)))

        # rotate 180
        imgs_rgb.append(F.to_tensor(F.rotate(rgb_img, 180)))
        imgs_uv.append(F.to_tensor(F.rotate(uv_img, 180)))

        # random rotate ±20°
        rot_rgb, rot_uv = self.paired_random_rotation(rgb_img, uv_img, self.rotation_degree)
        imgs_rgb.append(F.to_tensor(rot_rgb))
        imgs_uv.append(F.to_tensor(rot_uv))

        # random resized crop (scale, same crop for rgb and uv)
        crop_rgb, crop_uv = self.paired_random_resized_crop(rgb_img, uv_img)
        imgs_rgb.append(F.to_tensor(crop_rgb))
        imgs_uv.append(F.to_tensor(crop_uv))
        # print("增强后尺寸：", imgs_rgb[0].shape)  # torch.Size([3, 1088, 1088])

        return imgs_rgb, imgs_uv



# 用于验证集或测试集，只返回中心裁剪原图
class PairedEvalTransform:
    def __init__(self, resize_shorter=1088, output_size=(1088, 1088)):
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

        rgb_tensor = F.to_tensor(rgb_img)
        uv_tensor = F.to_tensor(uv_img)
        return rgb_tensor, uv_tensor
# import random
# from PIL import Image
# from torchvision import transforms
#
# class PairedTrainTransform:
#     def __init__(self, resize_shorter=2048, output_size=(2048, 2048), max_rotation=10):
#         self.resize_shorter = resize_shorter
#         self.output_size = output_size
#         self.max_rotation = max_rotation
#
#     def __call__(self, rgb, uv):
#         w, h = rgb.size
#         scale = self.resize_shorter / min(w, h)
#         new_w, new_h = int(w * scale), int(h * scale)
#         rgb = rgb.resize((new_w, new_h), Image.BILINEAR)
#         uv = uv.resize((new_w, new_h), Image.BILINEAR)
#
#         # 随机旋转
#         angle = random.uniform(-self.max_rotation, self.max_rotation)
#         rgb = rgb.rotate(angle, resample=Image.BILINEAR)
#         uv = uv.rotate(angle, resample=Image.BILINEAR)
#
#         # 随机裁剪（这里用随机位置裁剪）
#         crop_size = self.resize_shorter
#         i = random.randint(0, new_h - crop_size)
#         j = random.randint(0, new_w - crop_size)
#         rgb = rgb.crop((j, i, j + crop_size, i + crop_size))
#         uv = uv.crop((j, i, j + crop_size, i + crop_size))
#
#         # 随机水平翻转
#         if random.random() > 0.5:
#             rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
#             uv = uv.transpose(Image.FLIP_LEFT_RIGHT)
#
#         # 最后resize到目标尺寸
#         rgb = rgb.resize(self.output_size, Image.BILINEAR)
#         uv = uv.resize(self.output_size, Image.BILINEAR)
#
#         # 转成Tensor
#         rgb = transforms.ToTensor()(rgb)
#         uv = transforms.ToTensor()(uv)
#         return rgb, uv
#
#     class PairedEvalTransform:
#         def __init__(self, resize_shorter=2048, output_size=(2048, 2048), max_rotation=10):
#             self.resize_shorter = resize_shorter
#             self.output_size = output_size
#             self.max_rotation = max_rotation
#
#         def __call__(self, rgb, uv):
#             w, h = rgb.size
#             scale = self.resize_shorter / min(w, h)
#             new_w, new_h = int(w * scale), int(h * scale)
#             rgb = rgb.resize((new_w, new_h), Image.BILINEAR)
#             uv = uv.resize((new_w, new_h), Image.BILINEAR)
#
#             # 随机旋转
#             angle = random.uniform(-self.max_rotation, self.max_rotation)
#             rgb = rgb.rotate(angle, resample=Image.BILINEAR)
#             uv = uv.rotate(angle, resample=Image.BILINEAR)
#
#             # 随机裁剪（这里用随机位置裁剪）
#             crop_size = self.resize_shorter
#             i = random.randint(0, new_h - crop_size)
#             j = random.randint(0, new_w - crop_size)
#             rgb = rgb.crop((j, i, j + crop_size, i + crop_size))
#             uv = uv.crop((j, i, j + crop_size, i + crop_size))
#
#             # 随机水平翻转
#             if random.random() > 0.5:
#                 rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
#                 uv = uv.transpose(Image.FLIP_LEFT_RIGHT)
#
#             # 最后resize到目标尺寸
#             rgb = rgb.resize(self.output_size, Image.BILINEAR)
#             uv = uv.resize(self.output_size, Image.BILINEAR)
#
#             # 转成Tensor
#             rgb = transforms.ToTensor()(rgb)
#             uv = transforms.ToTensor()(uv)
#             return rgb, uv