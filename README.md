# Oxford-IIIT Pet 细粒度图像分类项目

基于 ResNet18 实现 37 类猫狗细粒度图像分类，包含模型训练、消融实验、混淆矩阵评估与 Grad-CAM 可解释性可视化。

## 项目简介

本项目使用 Oxford-IIIT Pet 数据集（共 7349 张图片，37 个猫狗品种），采用 ResNet18 作为基础卷积神经网络，完成细粒度图像分类任务。

- 数据集划分：训练集 5144 张，验证集 1100 张，测试集 1105 张
- 任务目标：对不同品种猫狗进行分类；使用 Grad-CAM 可视化模型关注区域，分析模型决策依据；通过消融实验对比标签平滑（Label Smoothing）对模型性能的影响。

## 环境依赖

```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 文件结构

```
pet_project/
├── data/              # 数据集存放目录
├── utils/             # 工具函数（评估、Grad-CAM）
├── dataset.py         # 数据集加载、划分、数据增强
├── model.py           # ResNet18 模型定义
├── train.py           # 模型训练、自动保存最优权重、绘制训练曲线
├── evaluate.py        # 模型评估、指标计算、混淆矩阵、Grad-CAM可视化
├── requirements.txt   # 依赖清单
└── README.md          # 项目说明与复现文档
```

## 一键复现命令（直接复制粘贴运行）

```
# 1. 安装项目依赖（清华源加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 基线模型训练（label_smoothing=0.0）
python train.py --label_smoothing 0.0

# 3. 消融实验训练（label_smoothing=0.1）
python train.py --label_smoothing 0.1

# 4. 模型评估，生成混淆矩阵、Grad-CAM热力图、指标结果
python evaluate.py
```

## 实验输出内容

运行完毕自动生成所有实验图片与模型权重：

- **training_curves.png**：训练 / 验证损失、准确率变化曲线
- **confusion_matrix.png**：原始混淆矩阵
- **confusion_matrix_normalized.png**：归一化混淆矩阵（报告主图）
- **gradcam_correct.png**：预测正确样本热力图
- **gradcam_wrong.png**：预测错误样本热力图
- **best_model.pth**：最优模型权重文件

## 实验结果汇总

本项目通过单变量消融实验验证标签平滑正则化效果，所有指标均为真实运行结果：

表格

| 实验配置                | Top-1 Acc(%) | Top-5 Acc(%) | Macro-F1 |
| ---                    | ---          | ---          | ---     |
| Baseline（smooth=0.0） | 91.30         | 99.09       | 0.9131   |
| Label-Smoothing=0.1    | 91.30         | 99.09        | 0.9131   |

## 实验结论

ResNet18 迁移学习在 Oxford-IIIT Pet 细粒度分类任务中表现良好，测试集准确率可达 91.30%。消融实验表明，本数据集基线模型过拟合程度较轻，因此 Label-Smoothing 正则化对整体分类指标提升效果有限。Grad-CAM 可视化证明模型多数情况下可聚焦宠物本体特征，复杂背景干扰是模型误判的主要原因。

## 复现说明

- 项目支持**完全一键复现**，所有数据、训练、评估、可视化全自动完成
- 推荐 GPU 环境运行，CPU 可运行但速度较慢
- 代码已固定随机种子，保证实验稳定性与可复现性