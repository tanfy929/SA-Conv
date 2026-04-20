import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class ResNetFPN_8_2(nn.Module):
    """
    ResNet+FPN, output resolution are 1/8 and 1/2.
    Each block has 2 layers.
    """

    def __init__(self, args):
        super().__init__()
        # Config
        block = BasicBlock
        initial_dim = args.initial_dim
        block_dims = args.block_dims

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(3, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8

        # 3. FPN upsample
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        # print('x shape in backbone:', x.shape) # ([16, 3, 48, 48]) ([1, 3, 1356, 2040])
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8

        # FPN
        x3_out = self.layer3_outconv(x3)

        # x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x3_out_2x = F.interpolate(x3_out, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

        # x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out_2x = F.interpolate(x2_out, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out+x2_out_2x) # ([8, 64, 128, 128])
        # return [x3_out, x1_out]
        return x1_out

# Attention Modules
def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1

# class ResNetFPN_16_4(nn.Module):
#     """
#     ResNet+FPN, output resolution are 1/16 and 1/4.
#     Each block has 2 layers.
#     """

#     def __init__(self, config):
#         super().__init__()
#         # Config
#         block = BasicBlock
#         initial_dim = config['initial_dim']
#         block_dims = config['block_dims']

#         # Class Variable
#         self.in_planes = initial_dim

#         # Networks
#         self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(initial_dim)
#         self.relu = nn.ReLU(inplace=True)

#         self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
#         self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
#         self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8
#         self.layer4 = self._make_layer(block, block_dims[3], stride=2)  # 1/16

#         # 3. FPN upsample
#         self.layer4_outconv = conv1x1(block_dims[3], block_dims[3])
#         self.layer3_outconv = conv1x1(block_dims[2], block_dims[3])
#         self.layer3_outconv2 = nn.Sequential(
#             conv3x3(block_dims[3], block_dims[3]),
#             nn.BatchNorm2d(block_dims[3]),
#             nn.LeakyReLU(),
#             conv3x3(block_dims[3], block_dims[2]),
#         )

#         self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
#         self.layer2_outconv2 = nn.Sequential(
#             conv3x3(block_dims[2], block_dims[2]),
#             nn.BatchNorm2d(block_dims[2]),
#             nn.LeakyReLU(),
#             conv3x3(block_dims[2], block_dims[1]),
#         )

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def _make_layer(self, block, dim, stride=1):
#         layer1 = block(self.in_planes, dim, stride=stride)
#         layer2 = block(dim, dim, stride=1)
#         layers = (layer1, layer2)

#         self.in_planes = dim
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         # ResNet Backbone
#         x0 = self.relu(self.bn1(self.conv1(x)))
#         x1 = self.layer1(x0)  # 1/2
#         x2 = self.layer2(x1)  # 1/4
#         x3 = self.layer3(x2)  # 1/8
#         x4 = self.layer4(x3)  # 1/16

#         # FPN
#         x4_out = self.layer4_outconv(x4)

#         x4_out_2x = F.interpolate(x4_out, scale_factor=2., mode='bilinear', align_corners=True)
#         x3_out = self.layer3_outconv(x3)
#         x3_out = self.layer3_outconv2(x3_out+x4_out_2x)

#         x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
#         x2_out = self.layer2_outconv(x2)
#         x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

#         return [x4_out, x2_out]

# class PositionEncodingSine(nn.Module):
#     """
#     This is a sinusoidal position encoding that generalized to 2-dimensional images
#     """

#     def __init__(self, d_model, max_shape=(256, 256), temp_bug_fix=True):
#         """
#         Args:
#             max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
#             temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
#                 the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
#                 on the final performance. For now, we keep both impls for backward compatability.
#                 We will remove the buggy impl after re-training all variants of our released models.
#         """
#         super().__init__()

#         pe = torch.zeros((d_model, *max_shape))
#         y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
#         x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
#         if temp_bug_fix:
#             div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
#         else:  # a buggy implementation (for backward compatability only)
#             div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2))
#         div_term = div_term[:, None, None]  # [C//4, 1, 1]
#         pe[0::4, :, :] = torch.sin(x_position * div_term)
#         pe[1::4, :, :] = torch.cos(x_position * div_term)
#         pe[2::4, :, :] = torch.sin(y_position * div_term)
#         pe[3::4, :, :] = torch.cos(y_position * div_term)

#         self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

#     def forward(self, x):
#         """
#         Args:
#             x: [N, C, H, W]
#         """
#         return x + self.pe[:, :, :x.size(2), :x.size(3)]
    
class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length
        KV = torch.einsum("nshd,nshv->nhdv", K, values)
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


# class LoFTREncoderLayer(nn.Module):
#     def __init__(self, d_model, nhead, attention='linear'):
#         super(LoFTREncoderLayer, self).__init__()

#         self.dim = d_model // nhead
#         self.nhead = nhead

#         # multi-head attention
#         self.q_proj = nn.Linear(d_model, d_model, bias=False)
#         self.k_proj = nn.Linear(d_model, d_model, bias=False)
#         self.v_proj = nn.Linear(d_model, d_model, bias=False)
#         self.attention = LinearAttention() if attention == 'linear' else None
#         self.merge = nn.Linear(d_model, d_model, bias=False)

#         # feed-forward network
#         self.mlp = nn.Sequential(
#             nn.Linear(d_model*2, d_model*2, bias=False),
#             nn.ReLU(True),
#             nn.Linear(d_model*2, d_model, bias=False),
#         )

#         # norm and dropout
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)

#     def forward(self, x, source, x_mask=None, source_mask=None):
#         bs = x.size(0)
#         query, key, value = x, source, source

#         # multi-head attention
#         query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)
#         key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)
#         value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
#         message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)
#         message = self.merge(message.view(bs, -1, self.nhead*self.dim))
#         message = self.norm1(message)

#         # feed-forward network
#         message = self.mlp(torch.cat([x, message], dim=2))
#         message = self.norm2(message)

#         return x + message

# class LocalFeatureTransformer(nn.Module):
    # def __init__(self, config):
    #     super(LocalFeatureTransformer, self).__init__()

    #     self.config = config
    #     self.d_model = config['d_model']
    #     self.nhead = config['nhead']
    #     self.layer_names = config['layer_names']
    #     encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
    #     self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
    #     self._reset_parameters()

    # def _reset_parameters(self):
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)

    # def forward(self, feat0, feat1, mask0=None, mask1=None):
    #     for layer, name in zip(self.layers, self.layer_names):
    #         if name == 'self':
    #             feat0 = layer(feat0, feat0, mask0, mask0)
    #             feat1 = layer(feat1, feat1, mask1, mask1)
    #         elif name == 'cross':
    #             feat0 = layer(feat0, feat1, mask0, mask1)
    #             feat1 = layer(feat1, feat0, mask1, mask0)
    #         else:
    #             raise KeyError

    #     return feat0, feat1

class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(SelfAttentionLayer, self).__init__()
        
        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C] 单图像特征序列
            mask (torch.Tensor): [N, L] (optional)
        """
        bs, L, C = x.shape
        # print('x shape in SelfAttentionLayer:', x.shape)  # ([8, 128*128, 64])
        query = self.q_proj(x).view(bs, L, self.nhead, self.dim)
        key = self.k_proj(x).view(bs, L, self.nhead, self.dim)
        value = self.v_proj(x).view(bs, L, self.nhead, self.dim)
        
        message = self.attention(query, key, value, q_mask=mask, kv_mask=mask)
        message = self.merge(message.view(bs, L, self.nhead*self.dim))
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, kv_size):
        super(CrossAttentionLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead
        self.kv_size = kv_size

        # 同时生成keys和values
        self.memory_generator = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model * kv_size * 2))

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C] 单图像特征序列
            mask (torch.Tensor): [N, L] (optional)
        """
        bs, L, C = x.shape
        # print('x shape in SelfAttentionLayer:', x.shape)  # ([8, 128*128, 64])
        global_feat = x.mean(dim=1)  # [N, C]
        memory_params = self.memory_generator(global_feat)
        memory_params = memory_params.view(bs, self.kv_size * 2, C)
        keys = memory_params[:, :self.kv_size, :]  
        values = memory_params[:, self.kv_size:, :]  

        query = self.q_proj(x).view(bs, L, self.nhead, self.dim)
        key = self.k_proj(keys).view(bs, self.kv_size, self.nhead, self.dim)
        value = self.v_proj(values).view(bs, self.kv_size, self.nhead, self.dim)

        message = self.attention(query, key, value, q_mask=mask, kv_mask=mask)
        message = self.merge(message.view(bs, L, self.nhead*self.dim))
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, kv_size, num_layers=4):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([CrossAttentionLayer(d_model, nhead, kv_size) for _ in range(num_layers)])
        
    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C] 
            mask (torch.Tensor): [N, L] (optional)
        """
        for layer in self.layers:
            x = layer(x, mask)
        
        return x

class HyperNet(nn.Module):
    def __init__(self, args):
        super(HyperNet, self).__init__()

        self.backbone = ResNetFPN_8_2(args)
        self.transformer = TransformerEncoder(
            d_model = args.d_model,
            nhead = args.nhead,
            kv_size = args.kv_size,
            num_layers = args.num_layers
        )
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # Regression head
        # self.regression_head = nn.Sequential(
        #     nn.Linear(args.d_model, 512),
        #     nn.ReLU(True),
        #     nn.Dropout(0.3),
        #     nn.Linear(512, 256),
        #     nn.ReLU(True),
        #     nn.Dropout(0.2),
        #     nn.Linear(256, 128),
        #     nn.ReLU(True),
        #     nn.Dropout(0.1),
        #     nn.Linear(128, 3))
        
        # self.regression_head = nn.Sequential(
        #     nn.Linear(args.d_model, 64),
        #     nn.ReLU(True),
        #     nn.Dropout(0.2),
        #     nn.Linear(64, 3))

        self.regression_head = nn.Sequential(
            nn.Linear(args.d_model, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(64, 3))

        # self.regression_head = nn.Sequential(
        #     nn.Linear(args.d_model, 256),
        #     nn.ReLU(True),
        #     nn.Dropout(0.3),
        #     nn.Linear(256, 128),
        #     nn.ReLU(True),
        #     nn.Dropout(0.2),
        #     nn.Linear(128, 64),
        #     nn.ReLU(True),
        #     nn.Dropout(0.1),
        #     nn.Linear(64, 3))
        
    def forward(self, x):
        feat_c = self.backbone(x)  
        # print('feat_c shape:', feat_c.shape)  # ([8, 64, 128, 128])

        # Flatten to sequence [N, HW, C]
        bs, c, h, w = feat_c.shape
        feat_c_flat = rearrange(feat_c, 'n c h w -> n (h w) c')
        
        feat_c_transformed = self.transformer(feat_c_flat)
        
        # Global average pooling over spatial dimensions
        feat_global = feat_c_transformed.mean(dim=1)  # [N, C]
        # feat_c_spatial = rearrange(feat_c_transformed, 'n (h w) c -> n c h w', h=h, w=w)
        # feat_global = self.global_pool(feat_c_spatial)  # [N, C, 1, 1]
        # feat_global = feat_global.view(bs, -1)  # [N, C]

        # Predict parameters
        params = self.regression_head(feat_global)
        # theta0, Cx, Cy
        return params[:,0], params[:,1], params[:,2]
