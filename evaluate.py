import os
import shutil
import random
import traceback

import numpy as np
import torch

from dataset import get_dataloaders
from model import get_model
from utils.metrics import evaluate_testset
from utils.gradcam import run_gradcam

# ========== 固定随机种子（评估阶段主要是保证 DataLoader 顺序一致） ==========
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, _, test_loader = get_dataloaders(batch_size=32)

model = get_model(37).to(device)
model.load_state_dict(torch.load("./best_model.pth", map_location=device))
model.eval()

# 获取类别名称（Oxford-IIIT Pet 的 37 个类别）
class_names = None
try:
    if hasattr(test_loader.dataset, 'dataset'):
        full_ds = test_loader.dataset.dataset
        if hasattr(full_ds, 'classes'):
            class_names = full_ds.classes
except Exception:
    pass

if class_names is None:
    print("[warn] 未取到 classes，混淆矩阵将使用数字标签")
    class_names = [str(i) for i in range(37)]

print(f"测试集样本数: {len(test_loader.dataset)}")

# 1. 测试集评估，生成混淆矩阵（内部已有 no_grad，安全）
test_acc, f1, _ = evaluate_testset(model, test_loader, device, class_names=class_names)
print(f"Test Acc {test_acc:.4f}, Macro-F1 {f1:.4f}")

# 2. 先跑一遍推理，拿到全部预测（用于挑样本 + 算 Top-5）
all_preds, all_labels, all_logits = [], [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        out = model(imgs.to(device))
        all_logits.append(out.float().cpu())
        all_preds.append(out.argmax(dim=1).cpu())
        all_labels.append(labels)

preds = torch.cat(all_preds)
labels_t = torch.cat(all_labels)
logits = torch.cat(all_logits)

# Top-5（报告表格需要这一列）
top5 = (logits.topk(5, dim=1).indices == labels_t[:, None]).any(dim=1).float().mean().item()
print(f"Top-5 {top5:.4f}")

# 3. Grad-CAM 可视化（注意：不要在 no_grad 中调用！）
target_layer = model.layer4[-1]

# 挑一张预测正确的 + 一张预测错误的
ok = (preds == labels_t)
i_ok = int(torch.nonzero(ok).flatten()[0])
bad = torch.nonzero(~ok).flatten()
i_bad = int(bad[0]) if len(bad) else i_ok

for tag, i in (("correct", i_ok), ("wrong", i_bad)):
    img, lab = test_loader.dataset[i]          # Subset，用的是确定性 transform
    run_gradcam(model, target_layer, img, device, true_label=int(lab))
    shutil.copy("./gradcam_result.png", f"./gradcam_{tag}.png")
    print(f"样本#{i} -> gradcam_{tag}.png | "
          f"pred={class_names[int(preds[i])]}, true={class_names[int(labels_t[i])]}")

print("已输出 confusion_matrix.png, confusion_matrix_normalized.png, "
      "gradcam_result.png, gradcam_correct.png, gradcam_wrong.png")