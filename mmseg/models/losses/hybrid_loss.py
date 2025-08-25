import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES

@LOSSES.register_module()
class HybridLoss(nn.Module):
    def __init__(self, num_classes=2, alpha=0.8, gamma=2.0, smooth=1e-6, class_weight=None,
                 loss_weight=1.0):
        """
        参数说明：
        num_classes: 类别数（背景+轨迹）
        alpha: Focal和Dice的混合权重
        gamma: Focal Loss的参数
        smooth: 平滑系数
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        
        # 方向场卷积核（注册为buffer保证设备一致性）
        self.register_buffer('sobel_x', torch.tensor(
            [[[[-1, 0, 1], 
               [-2, 0, 2], 
               [-1, 0, 1]]]],  # shape(1,1,3,3)
            dtype=torch.float32))
        
        self.register_buffer('sobel_y', torch.tensor(
            [[[[-1, -2, -1], 
               [ 0,  0,  0], 
               [ 1,  2,  1]]]],  # shape(1,1,3,3)
            dtype=torch.float32))

    def focal_loss(self, inputs, targets):
        """多分类Focal Loss"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        prob = F.softmax(inputs, dim=1)
        pt = prob.gather(1, targets.unsqueeze(1)).squeeze()  # 获取对应类别的概率
        
        # 动态alpha调整（背景类权重0.75，前景0.25）
        alpha = torch.where(targets==0, 0.75, 0.25).to(inputs.device)
        
        fl = alpha * (1 - pt) ** self.gamma * ce_loss
        return fl.mean()

    def dice_loss(self, inputs, targets):
        """多类Dice Loss带连续性约束"""
        inputs = F.softmax(inputs, dim=1)
        targets_onehot = F.one_hot(targets, self.num_classes).permute(0,3,1,2).float()
        
        # 计算各通道的Dice
        intersection = (inputs * targets_onehot).sum(dim=(2,3))
        union = inputs.sum(dim=(2,3)) + targets_onehot.sum(dim=(2,3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 仅计算前景类的连续性约束
        pred_traj = inputs[:,1,:,:].unsqueeze(1)  # 提取前景预测
        target_traj = targets_onehot[:,1,:,:].unsqueeze(1)
        
        # 方向场计算
        def get_gradient(img, kernel):
            return F.conv2d(img, kernel, padding=1, stride=1, groups=1)
        
        grad_x_pred = get_gradient(pred_traj, self.sobel_x)
        grad_y_pred = get_gradient(pred_traj, self.sobel_y)
        grad_x_target = get_gradient(target_traj, self.sobel_x)
        grad_y_target = get_gradient(target_traj, self.sobel_y)
        
        # 方向一致性计算（仅在前景区域）
        grad_cos = (grad_x_pred*grad_x_target + grad_y_pred*grad_y_target)
        grad_norm = (grad_x_pred**2 + grad_y_pred**2 + 1e-6).sqrt() * \
                    (grad_x_target**2 + grad_y_target**2 + 1e-6).sqrt()
        continuity = (1 - (grad_cos / grad_norm).mean())
        
        return 0.3 * continuity # 仅优化前景通道

    def forward(self, inputs, targets, **kwargs):
        # 输入校验
        assert inputs.shape[1] == self.num_classes, f"输入通道数{inputs.shape[1]}与类别数{self.num_classes}不匹配"
        
        fl = self.focal_loss(inputs, targets)
        dl = self.dice_loss(inputs, targets)
        return self.alpha * fl + (1 - self.alpha) * dl

@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(self, num_classes=2, alpha=0.8, gamma=2.0, smooth=1e-6, class_weight=None,
                 loss_weight=1.0):
        """
        参数说明：
        num_classes: 类别数（背景+轨迹）
        alpha: Focal和Dice的混合权重
        gamma: Focal Loss的参数
        smooth: 平滑系数
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.loss_weight = loss_weight
        
        # 方向场卷积核（注册为buffer保证设备一致性）
        self.register_buffer('sobel_x', torch.tensor(
            [[[-1,0,1], [-2,0,2], [-1,0,1]]], dtype=torch.float32))  # shape(1,3,3)
        self.register_buffer('sobel_y', torch.tensor(
            [[[-1,-2,-1], [0,0,0], [1,2,1]]], dtype=torch.float32))

    def focal_loss(self, inputs, targets):
        """多分类Focal Loss"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        prob = F.softmax(inputs, dim=1)
        pt = prob.gather(1, targets.unsqueeze(1)).squeeze()  # 获取对应类别的概率
        
        # 动态alpha调整（背景类权重0.75，前景0.25）
        alpha = torch.where(targets==0, 0.75, 0.25).to(inputs.device)
        
        fl = alpha * (1 - pt) ** self.gamma * ce_loss
        return fl.mean()

    def dice_loss(self, inputs, targets):
        """多类Dice Loss带连续性约束"""
        inputs = F.softmax(inputs, dim=1)
        targets_onehot = F.one_hot(targets, self.num_classes).permute(0,3,1,2).float()
        
        # 计算各通道的Dice
        intersection = (inputs * targets_onehot).sum(dim=(2,3))
        union = inputs.sum(dim=(2,3)) + targets_onehot.sum(dim=(2,3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 仅计算前景类的连续性约束
        pred_traj = inputs[:,1,:,:].unsqueeze(1)  # 提取前景预测
        target_traj = targets_onehot[:,1,:,:].unsqueeze(1)
        
        # 方向场计算
        def get_gradient(img, kernel):
            return F.conv2d(img, kernel, padding=1, groups=1)
        
        grad_x_pred = get_gradient(pred_traj, self.sobel_x)
        grad_y_pred = get_gradient(pred_traj, self.sobel_y)
        grad_x_target = get_gradient(target_traj, self.sobel_x)
        grad_y_target = get_gradient(target_traj, self.sobel_y)
        
        # 方向一致性计算（仅在前景区域）
        grad_cos = (grad_x_pred*grad_x_target + grad_y_pred*grad_y_target)
        grad_norm = (grad_x_pred**2 + grad_y_pred**2 + 1e-6).sqrt() * \
                    (grad_x_target**2 + grad_y_target**2 + 1e-6).sqrt()
        continuity = (1 - (grad_cos / grad_norm).mean())
        
        return 0.3 * continuity # 仅优化前景通道

    def forward(self, inputs, targets, weight=None, ignore_index=None, **kwargs):
        # 输入校验
        assert inputs.shape[1] == self.num_classes, f"输入通道数{inputs.shape[1]}与类别数{self.num_classes}不匹配"
        
        fl = self.focal_loss(inputs, targets)
        # dl = self.dice_loss(inputs, targets)
        return self.loss_weight * fl

# 使用示例
if __name__ == "__main__":
    # 模拟数据：batch_size=2, 256x256图像，两类输出
    inputs = torch.randn(2, 2, 256, 256)  # 模型原始输出（未softmax）
    targets = torch.randint(0, 2, (2, 256, 256))  # 类别标签（0:背景，1:轨迹）

    criterion = MultiClassTrajectoryLoss(num_classes=2, alpha=0.7)
    loss = criterion(inputs, targets)
    
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Components - Focal: {criterion.alpha*criterion.focal_loss(inputs,targets):.4f}, "
          f"Dice+Continuity: {(1-criterion.alpha)*criterion.dice_loss(inputs,targets):.4f}")
