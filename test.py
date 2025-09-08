import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score
from dataset import DualModalDataset
from model import DualBranchInceptionV3  # 用你实际的模型名替换
import os
from tranform import  PairedEvalTransform
from split import stratified_split_auto
# === 基本路径与类别定义（保持一致）===
base_dir = "/media/mmsys/1TB/ZLH/Dual/data"
rgb_root = os.path.join(base_dir, "rgb")
uv_root = os.path.join(base_dir, "uv")
categories = ["stable", "progressive"]
label_map = {cat: idx for idx, cat in enumerate(categories)}

# === 变换与 Dataset 构建（保持一致）===
_, _, test_pairs, _, _, test_labels = stratified_split_auto(base_dir)
# test_samples 是列表 [(rgb_path, uv_path, label)]
test_samples = [(rgb_path, uv_path, label) for (rgb_path, uv_path), label in zip(test_pairs, test_labels)]
test_transform = PairedEvalTransform(resize_shorter=1088, output_size=(1088, 1088))
test_dataset = DualModalDataset(rgb_root, uv_root, categories, label_map, transform=test_transform)
test_dataset.samples = test_samples  # ✅手动注入子集
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)

# === 模型加载 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model =DualBranchInceptionV3(
    num_classes=2,
    weight_path=None  # 不加载主干预训练，因为你要加载完整权重
).to(device)

# 替换为你保存的模型文件名
model.load_state_dict(torch.load("/media/mmsys/1TB/ZLH/Dual/five model_epoch1_sp0.98_pr0.96.pth"))
#/media/mmsys/1TB/ZLH/Dual/model_epoch32_sp0.91_pr0.85.pth
model.eval()

# === 测试循环 ===
all_preds = []
all_labels = []

with torch.no_grad():
    for rgb, uv, label in test_loader:
        rgb = rgb.squeeze(1).to(device)
        uv = uv.squeeze(1).to(device)
        label = label.to(device)

        outputs = model(rgb, uv)
        _, predicted = torch.max(outputs.data, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

# === 指标计算 ===
correct = sum([p == l for p, l in zip(all_preds, all_labels)])
total = len(all_labels)
# print(total)
accuracy = 100 * correct / total

cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
else:
    specificity = 0.0
# print("Confusion Matrix:")
# print(cm)
precision = precision_score(all_labels, all_preds, zero_division=0)

print(f" 测试集准确率 Accuracy: {accuracy:.2f}%")
print(f" 测试集 Specificity: {specificity:.4f}")
print(f" 测试集 Precision: {precision:.4f}")