from model import DualBranchInceptionV3
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_score
from torch.utils.tensorboard import SummaryWriter
import os
from dataset import DualModalDataset, FlattenedAugmentedDataset
from tranform import PairedTrainTransform, PairedEvalTransform
from torch.optim.lr_scheduler import ReduceLROnPlateau
from split import stratified_split_auto

# 基本路径和类别定义
base_dir = "data"  # 修改为相对路径
rgb_root = os.path.join(base_dir, "rgb")
uv_root = os.path.join(base_dir, "uv")
categories = ["stable", "progressive"]
label_map = {cat: idx for idx, cat in enumerate(categories)}

train_pairs, val_pairs, test_pairs, train_labels, val_labels, test_labels = stratified_split_auto(base_dir)
train_samples = [(rgb_path, uv_path, label) for (rgb_path, uv_path), label in zip(train_pairs, train_labels)]
val_samples = [(rgb_path, uv_path, label) for (rgb_path, uv_path), label in zip(val_pairs, val_labels)]

# 🔧 修复1: 降低分辨率，减少计算量
train_transform = PairedTrainTransform(resize_shorter=512, output_size=(512, 512))
val_transform = PairedEvalTransform(resize_shorter=512, output_size=(512, 512))

# Dataset 实例化
train_dataset = DualModalDataset(rgb_root, uv_root, categories, label_map, transform=train_transform)
train_dataset.samples = train_samples
flat_train_dataset = FlattenedAugmentedDataset(train_dataset)

val_dataset = DualModalDataset(rgb_root, uv_root, categories, label_map, transform=val_transform)
val_dataset.samples = val_samples

# 🔧 修复2: 减小batch size，提高训练稳定性
train_loader = DataLoader(flat_train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

print(f"训练集样本数: {len(flat_train_dataset)}")
print(f"验证集样本数: {len(val_dataset)}")

# 设备与模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DualBranchInceptionV3(
    num_classes=len(categories),
    weight_path=None  # 暂时不使用预训练权重，避免路径问题
).to(device)

# 🔧 修复3: 优化训练参数
criterion = nn.CrossEntropyLoss(label_smoothing=0.2)  # 增加label smoothing
optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-3)  # 降低学习率，增加正则化
num_epochs = 100

# 🔧 修复4: 使用更适合的学习率调度器
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# 🔧 修复5: 添加早停机制
best_val_loss = float('inf')
patience = 15
patience_counter = 0

writer = SummaryWriter(log_dir=os.path.join("runs_fixed", "dual_branch"))

print("开始训练...")
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for rgb_imgs, uv_imgs, labels in train_loader:
        rgb_imgs = rgb_imgs.to(device)
        uv_imgs = uv_imgs.to(device)
        labels = labels.to(device)

        outputs = model(rgb_imgs, uv_imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        
        # 🔧 修复6: 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

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
            # 🔧 修复7: 正确处理验证集数据维度
            if rgb_tensor.dim() == 5:  # [B, 1, C, H, W]
                rgb_tensor = rgb_tensor.squeeze(1)  # [B, C, H, W]
                uv_tensor = uv_tensor.squeeze(1)
            
            rgb_tensor = rgb_tensor.to(device)
            uv_tensor = uv_tensor.to(device)
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

    # 计算指标
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = precision_score(all_labels, all_preds, zero_division=0)
    else:
        specificity = 0
        precision = 0

    # 学习率调度
    scheduler.step(val_loss)

    # TensorBoard记录
    writer.add_scalar("Loss/train", train_loss, epoch + 1)
    writer.add_scalar("Loss/val", val_loss, epoch + 1)
    writer.add_scalar("Accuracy/train", train_acc, epoch + 1)
    writer.add_scalar("Accuracy/val", val_acc, epoch + 1)
    writer.add_scalar("Specificity/val", specificity, epoch + 1)
    writer.add_scalar("Precision/val", precision, epoch + 1)
    writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch + 1)

    # 🔧 修复8: 改进的模型保存策略和早停
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # 保存最佳模型
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'specificity': specificity,
            'precision': precision
        }, 'best_model_fixed.pth')
        print(f"✅ 保存最佳模型 - Epoch {epoch + 1}, Val Loss: {val_loss:.4f}")
    else:
        patience_counter += 1

    print(f"Epoch [{epoch + 1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% "
          f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}% "
          f"Specificity: {specificity:.4f} Precision: {precision:.4f} "
          f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    # 早停检查
    if patience_counter >= patience:
        print(f"🛑 早停触发 - 验证loss连续{patience}个epoch未改善")
        break

writer.close()
print("✅ 训练完成！最佳模型已保存为 'best_model_fixed.pth'")
print("📊 使用命令查看训练曲线: tensorboard --logdir runs_fixed --port=6006")
