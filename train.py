import time
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
import math

# 训练一个 epoch 的函数
def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes, lr_scheduler, print_freq, scaler,
                    criterion):
    model.train()
    running_loss = 0.0
    for i, (images, masks) in enumerate(data_loader):
        images, masks = images.to(device), masks.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, masks)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        running_loss += loss.item()

        if i % print_freq == 0:
            print(f"Epoch [{epoch}], Step [{i}/{len(data_loader)}], Loss: {loss.item()}")

    return running_loss / len(data_loader), optimizer.param_groups[0]["lr"]

# 学习率调度器的函数
def create_lr_scheduler(optimizer, num_steps, epochs, warmup=True, warmup_epochs=5, warmup_factor=1e-3):
    def lr_lambda(step):
        if warmup and step < warmup_epochs * num_steps:
            alpha = float(step) / (warmup_epochs * num_steps)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU6(inplace=True)  # 将ReLU替换为ReLU6
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=True, reduction=16):
        super(DoubleConv, self).__init__()
        self.conv1 = SeparableConv(in_channels, out_channels)
        self.conv2 = SeparableConv(out_channels, out_channels)

        # 如果 use_se 为 True，则使用 ECABlock 代替 SEBlock
        self.use_se = use_se
        if self.use_se:
            self.se_block = ECABlock(out_channels)  # 使用ECABlock替换SEBlock

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_se:
            x = self.se_block(x)  # 使用 ECABlock
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # [N, C, H, W]
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class self_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, bilinear=False, base_c=12, use_se=True):
        super(self_net, self).__init__()
        self.n_channels = in_channels
        self.n_classes = num_classes
        self.bilinear = bilinear

        # 在 DoubleConv 中使用 SEBlock
        self.inc = DoubleConv(in_channels, base_c, use_se=use_se)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.outc = OutConv(base_c, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        se = F.adaptive_avg_pool2d(x, 1)  # 全局平均池化，输出每个通道的全局特征
        se = F.relu(self.fc1(se))         # 通过全连接层进行降维
        se = torch.sigmoid(self.fc2(se))  # 通过全连接层进行升维，使用sigmoid调整权重
        return x * se                     # 将输入特征与SEBlock输出按通道相乘

class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        # 根据输入通道数自动计算卷积核大小
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化
        y = self.avg_pool(x)
        # 将通道维度调整为卷积输入的格式
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        # 调整回通道维度并应用sigmoid
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y  # 通道注意力应用于原始输入

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # 将 targets 转换为 Long 类型
        targets = targets.long()

        BCE_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-BCE_loss)  # 防止 log(0)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # 对输入使用 sigmoid 来处理二分类情况，如果是多分类需要 softmax
        inputs = torch.sigmoid(inputs)

        # 将 255（忽略值）替换为 0 或其他适合的值，以避免在 one_hot 中出错
        targets = torch.where(targets == self.ignore_index, torch.tensor(0).to(targets.device), targets)

        # 将 targets 转换为 Long 类型
        targets = targets.long()

        # 将 targets 转换为 one-hot 编码，以便匹配 inputs 的形状
        num_classes = inputs.size(1)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # 展平 inputs 和 targets 以计算 dice coefficient
        inputs = inputs.contiguous().view(-1)
        targets_one_hot = targets_one_hot.contiguous().view(-1)

        intersection = (inputs * targets_one_hot).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets_one_hot.sum() + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, dice_weight=0.5, ignore_index=-100):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        # 计算 Focal Loss
        focal_loss = self.focal_loss(inputs, targets)
        # 计算 Dice Loss
        dice_loss = self.dice_loss(inputs, targets)
        # 综合损失
        total_loss = (1 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss
        return total_loss

class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()

        # 设置路径为 'train' 或 'val'
        self.flag = "train" if train else "val"

        # 根据新的数据集结构设置路径
        img_dir = os.path.join(root, "images", self.flag)
        mask_dir = os.path.join(root, "masks", self.flag)

        # 检查路径是否存在
        assert os.path.exists(img_dir), f"Image path '{img_dir}' does not exist."
        assert os.path.exists(mask_dir), f"Mask path '{mask_dir}' does not exist."

        # 获取所有 JPG 图像文件名
        img_names = [i for i in os.listdir(img_dir) if i.endswith(".jpg")]
        self.img_list = [os.path.join(img_dir, i) for i in img_names]

        # 对应的掩码 PNG 文件路径
        self.mask_list = [os.path.join(mask_dir, i.replace(".jpg", ".png")) for i in img_names]

        # 检查掩码文件是否存在
        for mask in self.mask_list:
            if not os.path.exists(mask):
                raise FileNotFoundError(f"Mask file {mask} does not exist.")

        self.transforms = transforms

    def __getitem__(self, idx):
        # 加载 JPG 图像和 PNG 掩码
        img = np.array(Image.open(self.img_list[idx]).convert('RGB'))  # 转换为 NumPy 数组
        mask = np.array(Image.open(self.mask_list[idx]).convert('L'))  # 转换为 NumPy 数组，单通道灰度图

        # 应用数据增强 (Albumentations expects named arguments)
        if self.transforms is not None:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, masks = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_masks = cat_list(masks, fill_value=255)
        return batched_imgs, batched_masks

def cat_list(tensors, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))
    batch_shape = (len(tensors),) + max_size
    batched_tensors = tensors[0].new(*batch_shape).fill_(fill_value)
    for tensor, pad_tensor in zip(tensors, batched_tensors):
        pad_tensor[..., :tensor.shape[-2], :tensor.shape[-1]].copy_(tensor)
    return batched_tensors

# 评估函数
def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_images = 0
    total_time = 0

    with torch.no_grad():
        for images, masks in data_loader:
            start_time = time.time()
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            # 判断 outputs 是否是字典，如果是字典，则提取 'out' 键的值
            if isinstance(outputs, dict):
                outputs = outputs['out']  # 使用 'out' 键提取预测结果
            else:
                outputs = outputs  # 如果不是字典，直接使用 outputs 作为结果

            preds = outputs.argmax(dim=1)
            total_time += time.time() - start_time

            # 累积混淆矩阵
            confusion_matrix += compute_confusion_matrix(preds, masks, num_classes)
            total_images += 1

    # 计算 IoU 时排除背景
    iou_per_class = compute_iou_from_confusion_matrix(confusion_matrix, num_classes)

    # 只计算 Class 1、2、3 的 mIoU
    mIoU = np.nanmean(iou_per_class)

    # 计算 FPS
    fps = total_images / total_time

    # 输出 Class 1、2、3 的 IoU
    print(f"Class 1 IoU: {iou_per_class[0]:.3f}, Class 2 IoU: {iou_per_class[1]:.3f}, Class 3 IoU: {iou_per_class[2]:.3f}")
    print(f"mIoU (Class 1-3): {mIoU:.3f}, FPS: {fps:.3f}")

    return iou_per_class, mIoU, fps

# 创建模型
def create_model(num_classes, input_size=(3,200, 200)):
    model = self_net(in_channels=3, num_classes=4, base_c=12, use_se=True)
    device = torch.device('cuda:0')
    model = model.to(device)

    # 输出模型的参数量
    summary(model, input_size=input_size)
    return model

# 计算混淆矩阵
def compute_confusion_matrix(preds, masks, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_class in range(num_classes):
        for pred_class in range(num_classes):
            confusion_matrix[true_class, pred_class] = ((masks == true_class) & (preds == pred_class)).sum().item()
    return confusion_matrix

# 从混淆矩阵计算每类的 IoU
def compute_iou_from_confusion_matrix(confusion_matrix, num_classes):
    ious = []
    # 假设背景类的索引是 0，忽略第 0 类
    for i in range(1, num_classes):  # 从 1 开始
        intersection = confusion_matrix[i, i]
        union = confusion_matrix[i, :].sum() + confusion_matrix[:, i].sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # 忽略没有的类别
        else:
            ious.append(intersection / union)
    return ious



def get_transforms(height, width, train=True):
    if train:
        return A.Compose([
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),  # 随机改变伽马值
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10, 50), p=0.5),
            A.RandomScale(scale_limit=0.1, p=0.5),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),  # RGB通道偏移
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),  # 色相饱和度调整
            A.Resize(height=height, width=width, p=1.0),  # 调整到固定大小
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=height, width=width, p=1.0),  # 测试时仅调整大小和归一化
            A.Normalize(),
            ToTensorV2(),
        ])

# 添加测试代码部分
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_classes = args.num_classes + 1  # 包含背景类

    # 设置图像大小
    height, width =200, 200

    # 加载训练集和验证集
    train_transforms = get_transforms(height=height, width=width, train=True)
    val_transforms = get_transforms(height=height, width=width, train=False)

    train_dataset = DriveDataset("dataset", train=True, transforms=train_transforms)
    val_dataset = DriveDataset("dataset", train=False, transforms=val_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=4,
                                               shuffle=True,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=4,
                                             pin_memory=True)

    # 创建模型
    model = create_model(num_classes=num_classes).to(device)
    # # 定义优化器
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=args.lr, weight_decay=args.weight_decay
    # )
    #
    # # 使用组合损失函数
    # # 使用 Dice + Cross-Entropy 组合损失函数
    # criterion = DiceCrossEntropyLoss(dice_weight=0.7, ignore_index=255)  # 可以调整 dice_weight
    #
    # # 学习率调度器
    # scaler = torch.cuda.amp.GradScaler() if args.amp else None
    # lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    #
    # best_dice = 0.
    # for epoch in range(args.start_epoch, args.epochs):
    #     # 训练过程
    #     mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
    #                                     lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler, criterion=criterion)
    #
    #     # 验证集评估
    #     iou_per_class, mIoU, fps = evaluate(model, val_loader, device=device, num_classes=num_classes)
    #     print(f"验证集：Epoch {epoch}, mIoU: {mIoU:.3f}, FPS: {fps:.3f}")
    #
    #     if mIoU > best_dice:
    #         best_dice = mIoU
    #         torch.save(model.state_dict(), "model.pth")
    # # 加载训练好的最佳模型
    # model.load_state_dict(torch.load("model.pth"))
    # 定义优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay
    )

    # 使用组合损失函数
    criterion = CombinedLoss(alpha=1, gamma=2, dice_weight=0.7, ignore_index=255)

    # 学习率调度器
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    best_dice = 0.
    # 设置目标 IoU 阈值
    target_iou = [0.7013, 0.8947, 0.8444]#可根据具体情况调整也可以直接保存miou最高的值，如上面注释所示
    best_iou_per_class = [0.0, 0.0, 0.0]  # 记录每个类别的最佳 IoU

    for epoch in range(args.start_epoch, args.epochs):
        # 训练过程
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler,
                                        criterion=criterion)

        # 验证集评估
        iou_per_class, mIoU, fps = evaluate(model, val_loader, device=device, num_classes=num_classes)
        print(f"验证集：Epoch {epoch}, mIoU: {mIoU:.3f}, FPS: {fps:.3f}")

        # 检查三类 IoU 是否达到目标值
        save_model = False
        for i in range(3):  # 检查每个类别的 IoU
            if iou_per_class[i] >= target_iou[i] and iou_per_class[i] > best_iou_per_class[i]:
                best_iou_per_class[i] = iou_per_class[i]
                save_model = True  # 如果当前 IoU 更高，保存权重

        # 如果满足保存条件，保存当前模型权重
        if save_model:
            torch.save(model.state_dict(), f"save_weights/model_best_epoch_{epoch}.pth")
            print(f"满足条件，保存当前最佳模型权重：Epoch {epoch}, IoU: {iou_per_class}")
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="./dataset", help="DRIVE root")
    # exclude background
    parser.add_argument("--num-classes", default=3, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=800, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='model.pth', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)