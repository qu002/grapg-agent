import os
from sklearn.model_selection import train_test_split

def extract_num(filename):
    """提取编号，例如 N001.JPG -> 001"""
    return ''.join(filter(str.isdigit, os.path.splitext(filename)[0]))

def collect_pairs(rgb_dir, uv_dir):
    """从 RGB 和 UV 文件夹中收集匹配的图像对"""
    rgb_files = {extract_num(f): os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.lower().endswith(".jpg")}
    uv_files = {extract_num(f): os.path.join(uv_dir, f) for f in os.listdir(uv_dir) if f.lower().endswith(".jpg")}

    common_keys = sorted(set(rgb_files) & set(uv_files))
    pairs = [(rgb_files[k], uv_files[k]) for k in common_keys]
    return pairs

def stratified_split_auto(base_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # 自动从文件夹读取 stable/progressive 图像对
    stable_pairs = collect_pairs(
        rgb_dir=os.path.join(base_dir, "rgb/stable"),
        uv_dir=os.path.join(base_dir, "uv/stable")
    )
    progressive_pairs = collect_pairs(
        rgb_dir=os.path.join(base_dir, "rgb/progressive"),
        uv_dir=os.path.join(base_dir, "uv/progressive")
    )

    all_pairs = stable_pairs + progressive_pairs
    all_labels = [0] * len(stable_pairs) + [1] * len(progressive_pairs)

    # 划分测试集
    pairs_train_val, pairs_test, labels_train_val, labels_test = train_test_split(
        all_pairs, all_labels, test_size=test_ratio, stratify=all_labels, random_state=42
    )

    # 划分验证集
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    pairs_train, pairs_val, labels_train, labels_val = train_test_split(
        pairs_train_val, labels_train_val, test_size=val_ratio_adjusted, stratify=labels_train_val, random_state=42
    )

    return pairs_train, pairs_val, pairs_test, labels_train, labels_val, labels_test

if __name__ == "__main__":
    base_dir = "/media/mmsys/1TB/ZLH/Dual/data"

    train_pairs, val_pairs, test_pairs, train_labels, val_labels, test_labels = stratified_split_auto(base_dir)

    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

    stable_train = sum(1 for l in train_labels if l == 0)
    progressive_train = sum(1 for l in train_labels if l == 1)
    print(f"Train set - Stable: {stable_train}, Progressive: {progressive_train}")

    stable_val = sum(1 for l in val_labels if l == 0)
    progressive_val = sum(1 for l in val_labels if l == 1)
    print(f"Val set - Stable: {stable_val}, Progressive: {progressive_val}")

    stable_test = sum(1 for l in test_labels if l == 0)
    progressive_test = sum(1 for l in test_labels if l == 1)
    print(f"Test set - Stable: {stable_test}, Progressive: {progressive_test}")

    print("Sample pair:", train_pairs[0])
