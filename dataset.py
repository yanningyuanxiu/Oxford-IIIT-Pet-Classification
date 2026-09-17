import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# 固定随机种子，保证每次划分数据一模一样
SEED = 42

def get_dataloaders(batch_size=32):
    # 训练集：带随机增强
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 验证、测试集：绝对不能随机翻转、随机裁剪！！只用确定性变换
    val_test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 自动下载Oxford‑IIIT Pet数据集（只创建一次）
    full_dataset = datasets.OxfordIIITPet(root="./data", download=True, transform=None)
    
    # 获取标签（使用更安全的方式，不依赖私有属性）
    targets = [sample[1] for sample in full_dataset]

    # 70%训练，15%验证，15%测试 分层划分
    train_idx, temp_idx = train_test_split(
        list(range(len(full_dataset))),
        test_size=0.3, random_state=SEED, stratify=targets
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5, random_state=SEED, 
        stratify=[targets[i] for i in temp_idx]
    )

    # 创建带transform的数据集（避免重复实例化）
    train_dataset = datasets.OxfordIIITPet(root="./data", transform=train_transform)
    val_dataset = datasets.OxfordIIITPet(root="./data", transform=val_test_transform)
    test_dataset = datasets.OxfordIIITPet(root="./data", transform=val_test_transform)

    # 用Subset划分
    train_set = Subset(train_dataset, train_idx)
    val_set = Subset(val_dataset, val_idx)
    test_set = Subset(test_dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    tr_loader, va_loader, te_loader = get_dataloaders(32)
    images, labels = next(iter(tr_loader))
    print("输入batch形状：", images.shape)
    print(f"训练集大小：{len(tr_loader.dataset)}")
    print(f"验证集大小：{len(va_loader.dataset)}")
    print(f"测试集大小：{len(te_loader.dataset)}")