import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import get_model
from tqdm import tqdm
import os
import random
import numpy as np
import matplotlib.pyplot as plt

# ========== 固定随机种子，保证可复现 ==========
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 设备自动判断，优先GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

# 超参数
EPOCHS = 10
LR = 1e-4
BATCH_SIZE = 32

# 加载数据
train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE)

# 加载模型
model = get_model(num_classes=37).to(device)

# 损失函数、优化器、学习率调度器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
# 注意：不要传 verbose=True，torch>=2.0 已移除该参数，会直接 TypeError
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=2, factor=0.5
)

best_val_acc = 0.0
save_path = "./best_model.pth"

# 记录训练过程
train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(EPOCHS):
    # ==========训练阶段==========
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Train")
    for imgs, labels in pbar_train:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        train_correct += torch.sum(preds == labels.data)
        train_total += labels.size(0)
        pbar_train.set_postfix({"loss": loss.item()})

    train_acc = train_correct.float() / train_total
    avg_train_loss = train_loss / len(train_loader)

    # ==========验证阶段==========
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Val")
        for imgs, labels in pbar_val:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            val_correct += torch.sum(preds == labels.data)
            val_total += labels.size(0)
            pbar_val.set_postfix({"loss": loss.item()})

    val_acc = val_correct.float() / val_total
    avg_val_loss = val_loss / len(val_loader)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accs.append(train_acc.item())
    val_accs.append(val_acc.item())

    print(f"\n【Epoch {epoch+1}】")
    print(f"Train Loss:{avg_train_loss:.4f} | Train Acc:{train_acc:.4f}")
    print(f"Val Loss:{avg_val_loss:.4f} | Val Acc:{val_acc:.4f}")
    print(f"当前学习率: {optimizer.param_groups[0]['lr']:.2e}\n")

    scheduler.step(val_acc)

    # 用验证集选最优模型（不能用测试集选，否则泄漏）
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"✅保存最优模型，当前最高验证准确率：{best_val_acc:.4f}")

# ==========全部epoch跑完，在测试集评估==========
print("\n========== 在测试集上评估最终模型 ==========")
model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Test"):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, dim=1)
        test_correct += torch.sum(preds == labels.data)
        test_total += labels.size(0)

test_acc = test_correct.float() / test_total
print(f"测试集准确率 Test Acc: {test_acc:.4f}")

# ==========绘制训练曲线==========
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS+1), train_losses, 'b-', label='Train Loss')
plt.plot(range(1, EPOCHS+1), val_losses, 'r-', label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS+1), train_accs, 'b-', label='Train Acc')
plt.plot(range(1, EPOCHS+1), val_accs, 'r-', label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("./training_curves.png", dpi=300)
print("\n✅训练曲线已保存为 training_curves.png")