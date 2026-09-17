import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns

def evaluate_testset(model, test_loader, device, class_names=None):
   
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            out = model(imgs)
            _, preds = torch.max(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    acc = np.mean(np.array(all_preds) == np.array(all_labels))

    # 若未提供类别名称，则使用数字标签
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    # 绘制混淆矩阵（原始计数版）
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm.astype(int), cmap="Blues", annot=True, fmt="d",
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 8})
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Counts)")
    plt.tight_layout()
    plt.savefig("./confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 绘制归一化混淆矩阵（可选，用于报告更直观）
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm_normalized, cmap="Blues", annot=True, fmt=".2f",
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 8})
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Normalized)")
    plt.tight_layout()
    plt.savefig("./confusion_matrix_normalized.png", dpi=300, bbox_inches='tight')
    plt.close()

    return acc, macro_f1, cm