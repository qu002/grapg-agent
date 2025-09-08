"""
轻量化训练脚本 - 专为小数据集优化
使用轻量化模型和数据增强策略，解决过拟合问题
"""

from model_light import DualBranchLightweight, DualBranchMini, count_parameters
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
import os
from dataset import DualModalDataset, FlattenedAugmentedDataset
from transform_light import PairedLightTrainTransform, PairedMinimalTrainTransform
from tranform import PairedEvalTransform
from torch.optim.lr_scheduler import ReduceLROnPlateau
from split import stratified_split_auto
import time

def train_lightweight_model():
    # 配置参数
    config = {
        'base_dir': 'data',
        'input_size': 384,  # 进一步降低分辨率
        'batch_size': 8,    # 增加batch size提高稳定性
        'num_epochs': 80,
        'learning_rate': 1e-4,
        'weight_decay': 1e-3,
        'patience': 12,
        'model_type': 'lightweight',  # 'mini', 'lightweight', 'efficientnet'
        'augmentation': 'minimal'     # 'minimal', 'light', 'stable'
    }
    
    print("🚀 启动轻量化训练...")
    print(f"配置: {config}")
    
    # 基本设置
    base_dir = config['base_dir']
    rgb_root = os.path.join(base_dir, "rgb")
    uv_root = os.path.join(base_dir, "uv")
    categories = ["stable", "progressive"]
    label_map = {cat: idx for idx, cat in enumerate(categories)}
    
    # 数据分割
    train_pairs, val_pairs, test_pairs, train_labels, val_labels, test_labels = stratified_split_auto(base_dir)
    train_samples = [(rgb_path, uv_path, label) for (rgb_path, uv_path), label in zip(train_pairs, train_labels)]
    val_samples = [(rgb_path, uv_path, label) for (rgb_path, uv_path), label in zip(val_pairs, val_labels)]
    
    print(f"📊 数据分布:")
    print(f"  训练集: {len(train_samples)} 样本")
    print(f"  验证集: {len(val_samples)} 样本")
    print(f"  测试集: {len(test_pairs)} 样本")
    
    # 选择数据增强策略
    if config['augmentation'] == 'minimal':
        train_transform = PairedMinimalTrainTransform(
            resize_shorter=config['input_size'], 
            output_size=(config['input_size'], config['input_size'])
        )
        print("📈 使用最小化数据增强 (2倍)")
    else:
        train_transform = PairedLightTrainTransform(
            resize_shorter=config['input_size'], 
            output_size=(config['input_size'], config['input_size'])
        )
        print("📈 使用轻量化数据增强 (3倍)")
    
    val_transform = PairedEvalTransform(
        resize_shorter=config['input_size'], 
        output_size=(config['input_size'], config['input_size'])
    )
    
    # 数据集和数据加载器
    train_dataset = DualModalDataset(rgb_root, uv_root, categories, label_map, transform=train_transform)
    train_dataset.samples = train_samples
    flat_train_dataset = FlattenedAugmentedDataset(train_dataset)
    
    val_dataset = DualModalDataset(rgb_root, uv_root, categories, label_map, transform=val_transform)
    val_dataset.samples = val_samples
    
    train_loader = DataLoader(
        flat_train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
    print(f"🔄 训练样本总数: {len(flat_train_dataset)} (增强后)")
    print(f"🔄 验证样本总数: {len(val_dataset)}")
    
    # 模型选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备: {device}")
    
    if config['model_type'] == 'mini':
        model = DualBranchMini(num_classes=len(categories))
        print("🏗️  使用超轻量化模型 (DualBranchMini)")
    else:
        model = DualBranchLightweight(num_classes=len(categories))
        print("🏗️  使用轻量化模型 (DualBranchLightweight)")
    
    model = model.to(device)
    
    # 打印模型信息
    total_params = count_parameters(model)
    print(f"📊 模型参数量: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True,
        min_lr=1e-6
    )
    
    # 训练监控
    writer = SummaryWriter(log_dir=f"runs_lightweight/{config['model_type']}_{config['augmentation']}")
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    
    print("🎯 开始训练...")
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (rgb_imgs, uv_imgs, labels) in enumerate(train_loader):
            rgb_imgs = rgb_imgs.to(device)
            uv_imgs = uv_imgs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(rgb_imgs, uv_imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for rgb_tensor, uv_tensor, label_tensor in val_loader:
                if rgb_tensor.dim() == 5:
                    rgb_tensor = rgb_tensor.squeeze(1)
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
        
        # 计算详细指标
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 记录到TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Loss/val", val_loss, epoch + 1)
        writer.add_scalar("Accuracy/train", train_acc, epoch + 1)
        writer.add_scalar("Accuracy/val", val_acc, epoch + 1)
        writer.add_scalar("Metrics/precision", precision, epoch + 1)
        writer.add_scalar("Metrics/recall", recall, epoch + 1)
        writer.add_scalar("Metrics/f1", f1, epoch + 1)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch + 1)
        
        # 模型保存和早停
        improved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improved = True
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            improved = True
            
        if improved:
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }, f'best_model_{config["model_type"]}_{config["augmentation"]}.pth')
            status = "✅ 已保存"
        else:
            patience_counter += 1
            status = f"⏳ {patience_counter}/{config['patience']}"
        
        # 打印进度
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1:3d}/{config['num_epochs']}] "
              f"Train: {train_loss:.4f}/{train_acc:.1f}% "
              f"Val: {val_loss:.4f}/{val_acc:.1f}% "
              f"F1: {f1:.3f} LR: {optimizer.param_groups[0]['lr']:.1e} "
              f"Time: {elapsed/60:.1f}min {status}")
        
        # 早停检查
        if patience_counter >= config['patience']:
            print(f"🛑 早停触发 - 验证指标连续{config['patience']}个epoch未改善")
            break
    
    total_time = time.time() - start_time
    writer.close()
    
    print(f"✅ 训练完成!")
    print(f"⏱️  总训练时间: {total_time/60:.1f} 分钟")
    print(f"🏆 最佳验证准确率: {best_val_acc:.2f}%")
    print(f"🏆 最佳验证损失: {best_val_loss:.4f}")
    print(f"💾 模型已保存: best_model_{config['model_type']}_{config['augmentation']}.pth")
    print(f"📊 TensorBoard: tensorboard --logdir runs_lightweight --port=6007")

if __name__ == "__main__":
    train_lightweight_model()
