Lightweight U-Net for Semantic Segmentation
简介
这是一个深度学习项目，利用改进版 U-Net 模型实现语义分割任务。
本项目通过以下优化提升了模型的效率和性能：

轻量化设计：采用深度可分离卷积（Depthwise Separable Convolution）减少模型参数量。
增强特征表达：引入通道注意力机制（Efficient Channel Attention, ECA）。
自定义损失函数：结合 Focal Loss 和 Dice Loss，提升小目标检测能力并缓解类别不平衡问题。
灵活的数据增强：提供多种增强策略，如随机翻转、亮度调整等，提升模型对不同场景数据的适应性。
性能评估工具：包括交并比 (IoU)、平均交并比 (mIoU)、推理速度 (FPS) 等指标计算工具。
项目结构
├── dataset/              # 数据集文件夹
│   ├── images/           # 图像文件夹
│   │   ├── train/        # 训练集图像
│   │   ├── val/          # 验证集图像
│   ├── masks/            # 掩码文件夹
│       ├── train/        # 训练集掩码
│       ├── val/          # 验证集掩码
├── train.py              # 模型训练脚本
├── test.py               # 模型验证脚本
├── run.py                # 推理脚本，用于测试模型分割性能
├── README.md             # 项目文档
├── requirements.txt      # 依赖项文件
数据集说明
数据格式：支持 .jpg 和 .png 文件。
掩码说明：掩码图片中，每个像素值表示对应类别（如 0 表示背景，1 表示前景目标）。
dataset/
├── images/
│   ├── train/    # 训练集图像
│   ├── val/      # 验证集图像
├── masks/
│   ├── train/    # 训练集对应掩码
│   ├── val/      # 验证集对应掩码
安装依赖
请确保 Python 环境已安装以下依赖项：

bash
复制代码
pip install -r requirements.txt
参数说明：

--data-path：数据集路径。
--batch-size：每批次训练的图片数量（默认 16）。
--epochs：训练的总轮数（默认 50）。
--lr：学习率（默认 1e-4）。
--device：使用的设备（cuda 或 cpu）。
性能对比
以下是改进版 U-Net 的实验结果：

模型版本	mIoU	参数量	推理速度（FPS）
原始 U-Net	80.2%	31.2M	10.5 FPS
改进版 U-Net	83.7%	20.3M	14.2 FPS
