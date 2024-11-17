import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
import math
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





# 计算混淆矩阵
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

# 计算每个类别的 IoU
def per_class_iu(hist):
    print('Defect class IoU as follows:')
    print(np.diag(hist)[1:] / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist))[1:], 1))
    return np.diag(hist)[1:] / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist))[1:], 1)

# 计算每个类别的准确率
def per_class_PA(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)

# 计算 mIoU 和 mPA
def compute_mIoU(gt_dir, pred_dir):
    num_classes = 4  # 根据任务类别数
    print('Num classes', num_classes)
    hist = np.zeros((num_classes, num_classes))

    npy_name_list = [f.split('_')[1].split('.')[0] for f in os.listdir(pred_dir) if f.endswith('.npy')]

    gt_npy_files = [os.path.join(gt_dir, f"ground_truth_{x}.npy") for x in npy_name_list]
    pred_npy_files = [os.path.join(pred_dir, f"prediction_{x}.npy") for x in npy_name_list]

    for ind in range(len(gt_npy_files)):
        if not os.path.isfile(gt_npy_files[ind]):
            print(f"Ground truth file not found: {gt_npy_files[ind]}")
            continue

        if not os.path.isfile(pred_npy_files[ind]):
            print(f"Prediction file not found: {pred_npy_files[ind]}")
            continue

        pred = np.load(pred_npy_files[ind])
        label = np.load(gt_npy_files[ind])

        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                len(label.flatten()), len(pred.flatten()), gt_npy_files[ind],
                pred_npy_files[ind]))
            continue

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

    mIoUs = per_class_iu(hist)
    mPA = per_class_PA(hist)

    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 4)) +
          '; mPA: ' + str(round(np.nanmean(mPA) * 100, 4)))

    return mIoUs


# 计算 IoU
def get_iandu_array(pred, ann, classIdx: int):
    if isinstance(pred, torch.Tensor): pred = pred.numpy()
    if isinstance(ann, torch.Tensor): ann = ann.numpy()
    i = np.sum(np.logical_and(np.equal(pred, ann), np.equal(ann, classIdx)))
    u = np.sum(np.logical_or(np.equal(pred, classIdx), np.equal(ann, classIdx)))
    return i, u

# 计算多个类别的 IoU
def get_ious_dir(preds_dir: str, anns_dir: str):
    preds = sorted([os.path.join(preds_dir, p) for p in os.listdir(preds_dir) if p.endswith('.npy')])
    anns = sorted([os.path.join(anns_dir, a) for a in os.listdir(anns_dir) if a.endswith('.npy')])

    i1, u1, i2, u2, i3, u3 = 0, 0, 0, 0, 0, 0
    for pred_file, ann_file in zip(preds, anns):
        pred = np.load(pred_file)
        ann = np.load(ann_file)

        i, u = get_iandu_array(pred, ann, 1)
        i1, u1 = i1 + i, u1 + u

        i, u = get_iandu_array(pred, ann, 2)
        i2, u2 = i2 + i, u2 + u

        i, u = get_iandu_array(pred, ann, 3)
        i3, u3 = i3 + i, u3 + u

    return i1 / u1, i2 / u2, i3 / u3

# 定义数据集类
class DriveDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, transforms=None):
        super(DriveDataset, self).__init__()
        assert os.path.exists(img_dir), f"Image path '{img_dir}' does not exist."
        assert os.path.exists(mask_dir), f"Mask path '{mask_dir}' does not exist."

        img_names = [i for i in os.listdir(img_dir) if i.endswith(".jpg")]
        self.img_list = [os.path.join(img_dir, i) for i in img_names]
        self.mask_list = [os.path.join(mask_dir, i.replace(".jpg", ".png")) for i in img_names]

        for mask in self.mask_list:
            if not os.path.exists(mask):
                raise FileNotFoundError(f"Mask file {mask} does not exist.")

        self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        mask_path = self.mask_list[index]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transforms is not None:
            augmented = self.transforms(image=np.array(image), mask=np.array(mask))
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

# 保存 ground truth 和 prediction 的 .npy 文件
def save_npy_files(loader, model, device, output_pred_dir, output_gt_dir, img_list):
    model.eval()  # 切换模型到评估模式

    # 确保输出文件夹存在
    if not os.path.exists(output_pred_dir):
        os.makedirs(output_pred_dir)
    if not os.path.exists(output_gt_dir):
        os.makedirs(output_gt_dir)

    with torch.no_grad():  # 关闭梯度计算
        for idx, (image, mask) in enumerate(loader):
            image = image.to(device)
            mask = mask.to(device)

            # 模型预测
            output = model(image)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            # 使用 img_list 提取文件名（去掉扩展名）
            img_name = os.path.basename(img_list[idx]).split('.')[0]

            # 保存 ground truth 和预测的 .npy 文件
            gt_npy_path = os.path.join(output_gt_dir, f"ground_truth_{img_name}.npy")
            pred_npy_path = os.path.join(output_pred_dir, f"prediction_{img_name}.npy")

            np.save(gt_npy_path, mask.squeeze(0).cpu().numpy())  # 保存 ground truth
            np.save(pred_npy_path, pred)  # 保存预测结果

            print(f"Saved ground truth to {gt_npy_path} and prediction to {pred_npy_path}")

# 创建模型
def create_model(num_classes, input_size=(3, 200, 200)):
    model = self_net(in_channels=3, num_classes=num_classes, base_c=12)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model

# 获取数据集的 transforms
def get_transforms(height, width, train=False):
    return A.Compose([
        A.Resize(height=height, width=width, p=1.0),
        A.Normalize(),
        ToTensorV2()
    ])

# 主程序
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 4  # 包括背景类
    height, width = 200,200

    img_dir = "./dataset/images/test"
    mask_dir = "./dataset/masks/test"

    test_transforms = get_transforms(height=height, width=width, train=False)
    test_dataset = DriveDataset(img_dir=img_dir, mask_dir=mask_dir, transforms=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, pin_memory=True)

    # 创建模型并加载权重
    model = create_model(num_classes=num_classes)
    model.load_state_dict(torch.load("./model.pth"))

    # 保存 ground truth 和 prediction 为 .npy 文件
    output_pred_dir = './test_predictions'
    output_gt_dir = './test_ground_truths'

    save_npy_files(test_loader, model, device, output_pred_dir, output_gt_dir, test_dataset.img_list)

    # 计算并输出 IoU
    iou1, iou2, iou3 = get_ious_dir(output_gt_dir, output_pred_dir)
    print(f"Class 1 IoU: {iou1:.3f}, Class 2 IoU: {iou2:.3f}, Class 3 IoU: {iou3:.3f}")

    # 计算并输出 mIoU
    mIoUs = compute_mIoU(output_gt_dir, output_pred_dir)
    print(f"mIoU: {np.nanmean(mIoUs):.3f}")



if __name__ == '__main__':
    main()
