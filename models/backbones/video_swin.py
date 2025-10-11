"""
Video Swin Transformer Backbone

Implementación del Video Swin Transformer para extracción de features espacio-temporales.
Basado en el paper: "Video Swin Transformer" (Liu et al., 2022)

Este modelo utiliza shifted window attention en 3D para capturar dependencias
espacio-temporales de manera eficiente.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math
from einops import rearrange


class PatchEmbed3D(nn.Module):
    """
    Patch Embedding 3D para videos
    
    Convierte video (B, C, T, H, W) en secuencia de patches (B, N, D)
    """
    
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (2, 4, 4),
        in_channels: int = 3,
        embed_dim: int = 96
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Projection usando conv3d
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W)
        
        Returns:
            x: (B, T', H', W', D)
        """
        # Projection
        x = self.proj(x)  # (B, D, T', H', W')
        
        # Rearrange para LayerNorm
        x = rearrange(x, 'b d t h w -> b t h w d')
        x = self.norm(x)
        
        return x


class WindowAttention3D(nn.Module):
    """
    Window-based Multi-head Self Attention en 3D
    
    Implementa attention dentro de ventanas locales 3D para eficiencia.
    """
    
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wt, Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)
        )
        
        # Get relative position index
        coords_t = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_t, coords_h, coords_w], indexing='ij'))  # 3, Wt, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wt*Wh*Ww
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wt*Wh*Ww, Wt*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wt*Wh*Ww, Wt*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        
        relative_position_index = relative_coords.sum(-1)  # Wt*Wh*Ww, Wt*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        
        # Q, K, V projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Initialize relative position bias
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (num_windows*B, Wt*Wh*Ww, C)
            mask: (num_windows, Wt*Wh*Ww, Wt*Wh*Ww) or None
        """
        B_, N, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # Relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # Apply mask if provided
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


def window_partition(x, window_size):
    """
    Particiona tensor en ventanas 3D
    
    Args:
        x: (B, T, H, W, C)
        window_size: (Wt, Wh, Ww)
    
    Returns:
        windows: (num_windows*B, Wt, Wh, Ww, C)
    """
    B, T, H, W, C = x.shape
    Wt, Wh, Ww = window_size
    
    x = x.view(B, T // Wt, Wt, H // Wh, Wh, W // Ww, Ww, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, Wt, Wh, Ww, C)
    
    return windows


def window_reverse(windows, window_size, T, H, W):
    """
    Revierte la partición de ventanas
    
    Args:
        windows: (num_windows*B, Wt, Wh, Ww, C)
        window_size: (Wt, Wh, Ww)
        T, H, W: Dimensiones originales
    
    Returns:
        x: (B, T, H, W, C)
    """
    Wt, Wh, Ww = window_size
    B = int(windows.shape[0] / (T * H * W / Wt / Wh / Ww))
    
    x = windows.view(B, T // Wt, H // Wh, W // Ww, Wt, Wh, Ww, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, T, H, W, -1)
    
    return x


class SwinTransformerBlock3D(nn.Module):
    """
    Bloque Swin Transformer 3D
    
    Implementa Window Attention seguido de shifted Window Attention
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int] = (8, 7, 7),
        shift_size: Tuple[int, int, int] = (0, 0, 0),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        
        # Window attention
        self.attn = WindowAttention3D(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # Layer normalization
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        
        # Pad if needed
        pad_l = pad_t = pad_f = 0
        pad_r = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        pad_b = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_back = (self.window_size[0] - T % self.window_size[0]) % self.window_size[0]
        
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_f, pad_back))
        _, Tp, Hp, Wp, _ = x.shape
        
        # Cyclic shift
        if any(i > 0 for i in self.shift_size):
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                dims=(1, 2, 3)
            )
        else:
            shifted_x = x
        
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)
        
        # Window attention
        attn_windows = self.attn(x_windows)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Tp, Hp, Wp)
        
        # Reverse cyclic shift
        if any(i > 0 for i in self.shift_size):
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                dims=(1, 2, 3)
            )
        else:
            x = shifted_x
        
        # Remove padding
        if pad_r > 0 or pad_b > 0 or pad_back > 0:
            x = x[:, :T, :H, :W, :].contiguous()
        
        # Residual connection
        x = shortcut + self.drop_path(x)
        
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class Mlp(nn.Module):
    """Multi-layer Perceptron"""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        
        return output


class BasicLayer(nn.Module):
    """
    Basic layer del Video Swin Transformer
    
    Consiste en múltiples Swin Transformer blocks con downsample opcional
    """
    
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Tuple[int, int, int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        
        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if i % 2 == 0 else (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path
            )
            for i in range(depth)
        ])
        
        # Downsample
        self.downsample = downsample
    
    def forward(self, x):
        """
        Args:
            x: (B, T, H, W, C)
        """
        for blk in self.blocks:
            x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x


class PatchMerging(nn.Module):
    """
    Patch Merging Layer para downsampling
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(8 * dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape
        
        # Padding
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, T % 2))
        
        # Merge patches
        x0 = x[:, 0::2, 0::2, 0::2, :]  # (B, T/2, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x4 = x[:, 0::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # (B, T/2, H/2, W/2, 8*C)
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x


class VideoSwinTransformer(nn.Module):
    """
    Video Swin Transformer completo
    
    Args:
        patch_size: Tamaño de patch (temporal, height, width)
        in_channels: Número de canales de entrada (RGB = 3)
        embed_dim: Dimensión de embedding inicial
        depths: Profundidad de cada stage
        num_heads: Número de attention heads en cada stage
        window_size: Tamaño de ventana (temporal, height, width)
        mlp_ratio: Ratio de expansión del MLP
        qkv_bias: Si usar bias en QKV projection
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Stochastic depth rate
        num_classes: Número de clases (0 = sin head de clasificación)
    """
    
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (2, 4, 4),
        in_channels: int = 3,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        window_size: Tuple[int, int, int] = (8, 7, 7),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        num_classes: int = 0
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        
        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None
            )
            self.layers.append(layer)
        
        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        
        # Classification head
        if num_classes > 0:
            self.head = nn.Linear(self.num_features, num_classes)
        else:
            self.head = None
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W)
        
        Returns:
            features: (B, T', H', W', D) si num_classes=0
            logits: (B, num_classes) si num_classes>0
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, T', H', W', D)
        x = self.pos_drop(x)
        
        # Apply layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # Si es para clasificación
        if self.head is not None:
            x = rearrange(x, 'b t h w d -> b d t h w')
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.head(x)
        
        return x
    
    def get_feature_maps(self, x):
        """
        Extrae feature maps sin pooling para TAD
        
        Returns:
            features: (B, T', H', W', D)
        """
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        return x


def video_swin_tiny(pretrained=False, **kwargs):
    """Video Swin Transformer Tiny"""
    model = VideoSwinTransformer(
        patch_size=(2, 4, 4),
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(8, 7, 7),
        **kwargs
    )
    return model


def video_swin_small(pretrained=False, **kwargs):
    """Video Swin Transformer Small"""
    model = VideoSwinTransformer(
        patch_size=(2, 4, 4),
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(8, 7, 7),
        **kwargs
    )
    return model


def video_swin_base(pretrained=False, **kwargs):
    """Video Swin Transformer Base"""
    model = VideoSwinTransformer(
        patch_size=(2, 4, 4),
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        window_size=(8, 7, 7),
        **kwargs
    )
    return model


if __name__ == '__main__':
    # Test del modelo
    print("Testing Video Swin Transformer...")
    
    # Crear modelo
    model = video_swin_tiny(num_classes=0)
    
    # Input dummy
    batch_size = 2
    video = torch.randn(batch_size, 3, 64, 224, 224)
    
    print(f"Input shape: {video.shape}")
    
    # Forward pass
    features = model.get_feature_maps(video)
    print(f"Output features shape: {features.shape}")
    
    # Contar parámetros
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Número de parámetros: {num_params:,}")
    
    # Test con clasificación
    model_cls = video_swin_tiny(num_classes=24)
    logits = model_cls(video)
    print(f"Classification logits shape: {logits.shape}")
    
    print("✓ Video Swin Transformer test completado")
