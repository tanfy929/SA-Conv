import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, mlp_hidden=16):
        """
        用于生成卷积核仿射形变参数的超网络
        :param in_channels: 输入图像通道数 (通常为3)
        :param out_channels: CNN特征提取后的通道数 N (即 p^2)
        :param mlp_hidden: MLP中间层维度
        """
        super(HyperNetwork, self).__init__()
        
        # 灰度化权重 (Rec. 601 标准)，设为不可训练的 buffer
        self.register_buffer('rgb_weights', torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1))
        
        # 轻量级多尺度特征提取器 (加入下采样)
        # 假设输入为 H x W，输出空间维度将降至 (H/8) x (W/8)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, out_channels, kernel_size=3, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # MLP：从 p^2 映射到最终的 3 个仿射参数
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, mlp_hidden//2),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden//2, 3)
        )
        # 将最后一层全连接层的 weight 和 bias 初始化为 0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. 灰度化处理 -> 形状: (B, 1, H, W)
        if C == 3:
            x_gray = (x * self.rgb_weights).sum(dim=1, keepdim=True)
        elif C == 1:
            x_gray = x
        else:
            raise ValueError(f"Expected 1 or 3 input channels, got {C}")

        # 2. 提取多尺度特征 F -> 形状: (B, p^2, H', W')
        F_feat = self.feature_extractor(x_gray)
        _, N, H_prime, W_prime = F_feat.shape
        
        # 3. 将输入 x_gray 下采样至与 F 相同的空间尺寸
        # 使用自适应平均池化保证严格对齐 -> 形状: (B, 1, H', W')
        x_down = F.adaptive_avg_pool2d(x_gray, (H_prime, W_prime))
        
        # 4. Reshape 操作准备内积投影
        # Z 形状: (B, N, H'*W')
        Z = F_feat.view(B, N, -1)
        # X 形状: (B, H'*W', 1)
        X_prime = x_down.view(B, -1, 1)
        
        # 5. 投影与均值化 (Z 矩阵乘 X_prime) -> w1 形状: (B, N, 1)
        w1 = torch.bmm(Z, X_prime) / (H_prime * W_prime)
        
        # 降维去除最后的维度 -> w1 形状: (B, N)
        w1 = w1.squeeze(-1) 
        
        # 6. 通过 MLP 得到最终的 3 个仿射参数 -> 形状: (B, 3)
        w = self.mlp(w1)
        
        return w

