"""
PINN模型 - 物理信息神经网络
支持 1 通道或 3 通道输入
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierConv2d(nn.Module):
    """傅里叶卷积层"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.local_conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size,
            padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        # 通道匹配（当 in_channels != out_channels 时）
        self.channel_match = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        x_fft = torch.fft.rfft2(x, norm='ortho')
        x_real = x_fft.real
        x_imag = x_fft.imag
        
        x_spectral = torch.fft.irfft2(torch.complex(x_real, x_imag), s=(H, W), norm='ortho')
        x_spectral = self.channel_match(x_spectral)  # 匹配通道数
        x_local = self.local_conv(x_spectral)
        out = x_spectral + x_local
        out = F.relu(self.bn(out))
        
        return out


class PhysicsBlock(nn.Module):
    """物理约束块：提取频率特征"""
    
    def __init__(self, in_channels, out_channels=16, conv_type='fourier'):
        super().__init__()
        if conv_type == 'fourier':
            self.conv1 = FourierConv2d(in_channels, out_channels)
            self.conv2 = FourierConv2d(out_channels, out_channels)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DepthwiseSeparableResidualBlock(nn.Module):
    """深度可分离残差块"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride,
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False)
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
        self.dw2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1,
                      groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
        self.pw2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.pw(self.dw(x))
        out = self.pw2(self.dw2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, D = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class PINNFaultDiagnosis(nn.Module):
    """
    物理信息神经网络故障诊断模型
    支持 1 通道或 3 通道输入
    """
    
    def __init__(self, in_channels=1, num_classes=10, 
                 conv_type='fourier', use_mha=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # 物理约束块
        self.physics_block = PhysicsBlock(in_channels, 16, conv_type=conv_type)
        
        # CNN特征提取
        self.layer1 = self._make_layer(16, 32, 2, stride=2)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.layer4 = self._make_layer(128, 256, 2, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 多头自注意力
        self.use_mha = use_mha
        if use_mha:
            self.mha = MultiHeadSelfAttention(dim=256, num_heads=8, dropout=0.1)
            self.mha_norm = nn.LayerNorm(256)
        
        # 分类器
        self.fc = nn.Linear(256, num_classes)
        
        # 物理约束参数
        self.freq_weight = nn.Parameter(torch.ones(1, 16, 1, 1))
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(DepthwiseSeparableResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(DepthwiseSeparableResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def physics_loss(self, features):
        """物理约束损失"""
        freq_energy = torch.fft.fft2(features)
        freq_energy = torch.abs(freq_energy)
        
        diff_h = torch.diff(freq_energy, dim=2)
        diff_w = torch.diff(freq_energy, dim=3)
        
        smoothness_loss = torch.mean(diff_h ** 2) + torch.mean(diff_w ** 2)
        
        return smoothness_loss
    
    def extract_features(self, x):
        """提取特征"""
        physics_features = self.physics_block(x)
        physics_features = physics_features * self.freq_weight
        x = self.layer1(physics_features)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if self.use_mha:
            B, C, H, W = x.shape
            x_tokens = x.flatten(2).transpose(1, 2)
            x_attn = self.mha(x_tokens)
            x_attn = x_tokens + x_attn
            x_attn = self.mha_norm(x_attn)
            x = x_attn.transpose(1, 2).reshape(B, C, H, W)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x, physics_features
    
    def forward(self, x):
        physics_features = self.physics_block(x)
        physics_features = physics_features * self.freq_weight
        
        x = self.layer1(physics_features)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if self.use_mha:
            B, C, H, W = x.shape
            x_tokens = x.flatten(2).transpose(1, 2)
            x_attn = self.mha(x_tokens)
            x_attn = x_tokens + x_attn
            x_attn = self.mha_norm(x_attn)
            x = x_attn.transpose(1, 2).reshape(B, C, H, W)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        
        return logits, physics_features
