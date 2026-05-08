import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperNetwork(nn.Module):
    def __init__(self, in_channels=35, out_channels=32, mlp_hidden=16):
        """
        用于生成卷积核仿射形变参数的超网络 (适配高维特征输入)
        :param in_channels: 输入特征的通道数 (如去雨网络中的 32+3=35)
        :param out_channels: CNN特征提取后的通道数 N (即 p^2)
        :param mlp_hidden: MLP中间层维度
        """
        super(HyperNetwork, self).__init__()
        
        # 轻量级多尺度特征提取器
        # 将第一层的输入通道设为动态的 in_channels
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1),
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
        
        # MLP：从 out_channels 映射到最终的 3 个仿射参数
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, mlp_hidden//2),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden//2, 3)
        )
        
        # 将最后一层全连接层的 weight 和 bias 初始化为 0 (保证初始退化为 TL)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. 结构特征压缩 (替代原来的灰度化)
        # 沿通道维度求平均，将 C 通道压缩为 1 通道，作为投影的空间结构参考 -> 形状: (B, 1, H, W)
        x_ref = torch.mean(x, dim=1, keepdim=True) 

        # 2. 提取多尺度特征 F -> 形状: (B, out_channels, H', W')
        F_feat = self.feature_extractor(x)
        _, N, H_prime, W_prime = F_feat.shape
        
        # 3. 将参考图 x_ref 下采样至与 F_feat 相同的空间尺寸
        # -> 形状: (B, 1, H', W')
        x_down = F.adaptive_avg_pool2d(x_ref, (H_prime, W_prime))
        
        # 4. Reshape 操作准备内积投影
        # Z 形状: (B, N, H'*W')
        Z = F_feat.view(B, N, -1)
        # X_prime 形状: (B, H'*W', 1)
        X_prime = x_down.view(B, -1, 1)
        
        # 5. 投影与均值化 (Z 矩阵乘 X_prime) -> w1 形状: (B, N, 1)
        w1 = torch.bmm(Z, X_prime) / (H_prime * W_prime)
        
        # 降维去除最后的维度 -> w1 形状: (B, N)
        w1 = w1.squeeze(-1) 
        
        # 6. 通过 MLP 得到最终的 3 个仿射参数 -> 形状: (B, 3)
        w = self.mlp(w1)
        
        return w