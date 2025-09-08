from model import  DualBranchInceptionV3
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_score
from torch.utils.tensorboard import SummaryWriter
import os
from dataset import DualModalDataset,FlattenedAugmentedDataset
from tranform import PairedTrainTransform, PairedEvalTransform
from torch.optim.lr_scheduler import CosineAnnealingLR
from split import stratified_split_auto
# 基本路径和类别定义
base_dir = "/media/mmsys/1TB/ZLH/Dual/data"
rgb_root = os.path.join(base_dir, "rgb")
uv_root = os.path.join(base_dir, "uv")
categories = ["stable", "progressive"]  # 你的类别名列表
label_map = {cat: idx for idx, cat in enumerate(categories)}  # 类别到数字标签映射
train_pairs, val_pairs, test_pairs, train_labels, val_labels, test_labels = stratified_split_auto(base_dir)
train_samples = [(rgb_path, uv_path, label) for (rgb_path, uv_path), label in zip(train_pairs, train_labels)]
val_samples = [(rgb_path, uv_path, label) for (rgb_path, uv_path), label in zip(val_pairs, val_labels)]

# 数据增强与预处理
train_transform = PairedTrainTransform(resize_shorter=1088, output_size=(1088, 1088))
val_transform = PairedEvalTransform(resize_shorter=1088, output_size=(1088, 1088))
# Dataset 实例化
train_dataset = DualModalDataset(rgb_root, uv_root, categories, label_map, transform=train_transform)
train_dataset.samples = train_samples  # 注入训练样本
flat_train_dataset = FlattenedAugmentedDataset(train_dataset)
val_dataset = DualModalDataset(rgb_root, uv_root, categories, label_map, transform=val_transform)
val_dataset.samples = val_samples  # 注入验证样本
# DataLoader
train_loader = DataLoader(flat_train_dataset, batch_size=6, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=4, pin_memory=True)
# print(f"训练集样本数: {len(train_dataset)}")
# print(f"验证集样本数: {len(val_dataset)}")
# 设备与模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model =DualBranchInceptionV3(
    num_classes=len(categories),
    weight_path="/media/mmsys/1TB/ZLH/Dual/timm_inception_v3/model.safetensors"
    #  weight_path="/media/mmsys/1TB/ZLH/Dual/models--timm--resnet18.a1_in1k/model.safetensors"
).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=1e-4,weight_decay=1e-4)
num_epochs = 150
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

writer = SummaryWriter(log_dir=os.path.join("runsix", "dual_branch"))

best_val_acc = 0.0
best_specificity = 0.0
best_precision = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_augmented_samples = 0  # 新增计数器

    for rgb_imgs, uv_imgs, labels in train_loader:
        rgb_imgs = rgb_imgs.to(device)  # [B, C, H, W]
        uv_imgs = uv_imgs.to(device)
        labels = labels.to(device)

        outputs = model(rgb_imgs, uv_imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    # print(f"Epoch {epoch + 1} 训练集增强样本总数: {total_augmented_samples}")
    train_acc = 100 * correct / total
    train_loss = running_loss / len(train_loader)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for rgb_tensor, uv_tensor, label_tensor in val_loader:
            rgb_tensor = rgb_tensor.squeeze(1).to(device)  # 去掉n_aug维度
            uv_tensor = uv_tensor.squeeze(1).to(device)
            label_tensor = label_tensor.to(device)

            outputs = model(rgb_tensor, uv_tensor)
            loss = criterion(outputs, label_tensor)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += label_tensor.size(0)
            val_correct += (predicted == label_tensor).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label_tensor.cpu().numpy())

    val_acc = 100 * val_correct / val_total
    val_loss /= len(val_loader)
    # total = len(all_labels)
    # print(total)
    # 计算 Specificity 和 Precision
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
        fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
        fn = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
        tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = precision_score(all_labels, all_preds, zero_division=0)

    # TensorBoard 记录
    writer.add_scalar("Loss/train", train_loss, epoch + 1)
    writer.add_scalar("Loss/val", val_loss, epoch + 1)
    writer.add_scalar("Accuracy/train", train_acc, epoch + 1)
    writer.add_scalar("Accuracy/val", val_acc, epoch + 1)
    writer.add_scalar("Specificity/val", specificity, epoch + 1)
    writer.add_scalar("Precision/val", precision, epoch + 1)

    # 保存最佳模型权重示例（可根据指标灵活修改）
    if specificity >= 0.80 and precision >= 0.80:
        model_path = f"six model_epoch{epoch + 1}_sp{specificity:.2f}_pr{precision:.2f}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Saved six model at epoch {epoch + 1} with specificity {specificity:.2f} and precision {precision:.2f}")

    print(f"six Epoch [{epoch + 1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% "
          f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}% "
          f"Val Specificity: {specificity:.4f} Val Precision: {precision:.4f} ")

    scheduler.step()

writer.close()
print("所有满足条件的模型权重已保存。")
# tensorboard --logdir runs --port=6008
