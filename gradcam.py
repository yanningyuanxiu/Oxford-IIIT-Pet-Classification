import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class _GradCAM:
    """极简 Grad-CAM：前向存特征，反向存梯度。"""

    def __init__(self, target_layer):
        self.act = None
        self.grad = None
        self.fh = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'act', o.detach()))
        if hasattr(target_layer, "register_full_backward_hook"):
            self.bh = target_layer.register_full_backward_hook(
                lambda m, gi, go: setattr(self, 'grad', go[0].detach()))
        else:  # torch < 1.8
            self.bh = target_layer.register_backward_hook(
                lambda m, gi, go: setattr(self, 'grad', go[0].detach()))

    def remove(self):
        self.fh.remove()
        self.bh.remove()


def run_gradcam(model, target_layer, img_tensor, device, true_label=None):
    """
    img_tensor: [3, H, W] 已归一化的 tensor
    ⚠️ 调用处绝不能包在 torch.no_grad() / torch.inference_mode() 下
    返回: (cam numpy [H,W], pred_class)
    """
    model.to(device)
    model.eval()                                   # eval 可以，但绝不用 no_grad

    x = img_tensor.unsqueeze(0).to(device).clone().requires_grad_(True)

    # 先取预测类别（不需要梯度）
    with torch.no_grad():
        pred_class = int(model(x).argmax(dim=1).item())

    g = _GradCAM(target_layer)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits = logits.float()
        logits[:, pred_class].sum().backward()

        if g.act is None or g.grad is None:
            raise RuntimeError(
                "未捕获到特征/梯度。常见原因：① 调用被 no_grad 包裹；"
                "② target_layer 选错，前向没经过它；③ 该层输出被 detach 打断。"
            )

        # 全局平均池化求权重 -> 加权求和 -> ReLU -> 归一化
        w = g.grad[0].mean(dim=(1, 2))
        cam = torch.relu((w[:, None, None] * g.act[0]).sum(dim=0))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
    finally:
        g.remove()

    # 上采样到原图尺寸
    H, W = x.shape[-2:]
    cam = F.interpolate(cam[None, None].float(), size=(H, W),
                        mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

    # 反归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    base = (img_tensor.detach().float().cpu().unsqueeze(0) * std + mean).clamp(0, 1)
    base = base[0].permute(1, 2, 0).numpy()

    # 三联图
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(base)
    axes[0].set_title("Input")
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(base)
    axes[2].imshow(cam, cmap="jet", alpha=0.5)
    axes[2].set_title("Overlay")

    title = f"Pred: {pred_class}"
    if true_label is not None:
        title += f", True: {true_label}"
    fig.suptitle(title, fontsize=11)
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("./gradcam_result.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[gradcam] 已保存 ./gradcam_result.png | pred={pred_class} true={true_label}")

    return cam, pred_class