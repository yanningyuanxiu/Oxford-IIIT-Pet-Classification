import torch
import torchvision
import torch.nn as nn

def get_model(num_classes=37, freeze_backbone=False):
  
    assert num_classes > 0
    
    # 加载预训练ResNet18
    model = torchvision.models.resnet18(weights="DEFAULT")
    
    # 可选：冻结主干网络
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # 修改最后一层，输出指定类别数
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

if __name__ == "__main__":
    m = get_model(num_classes=37)
    x = torch.randn(2, 3, 224, 224)
    out = m(x)
    print("模型输出shape:", out.shape)
    
    # 测试冻结模式
    m_frozen = get_model(num_classes=37, freeze_backbone=True)
    frozen_params = sum(1 for p in m_frozen.parameters() if p.requires_grad)
    total_params = sum(1 for p in m_frozen.parameters())
    print(f"冻结模式：可训练参数 {frozen_params}/{total_params}")