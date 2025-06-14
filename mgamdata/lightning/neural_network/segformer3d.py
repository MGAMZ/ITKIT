"""
Standalone SegFormer3D implementation for 3D medical image segmentation.

@InProceedings{Perera_2024_CVPR,
    author    = {Perera, Shehan and Navard, Pouyan and Yilmaz, Alper},
    title     = {SegFormer3D: An Efficient Transformer for 3D Medical Image Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {4981-4988}
}

This is a standalone PyTorch implementation that can be used with any training framework.
Modified to support arbitrary DWH input, optimized using SDPA, and follows modern Python 3.12 conventions.
"""

from collections.abc import Sequence
import torch
import math
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Any


class SelfAttention(nn.Module):
    """Self-attention module with optional spatial reduction for SegFormer3D.
    
    This module implements multi-head self-attention with optional spatial reduction
    via convolution (sr_ratio > 1) and supports Scaled Dot-Product Attention (SDPA).
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        sr_ratio: int | tuple[int, int, int] | None = None,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        use_sdpa: bool = True,
    ) -> None:
        """Initialize self-attention module.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            sr_ratio: Spatial reduction ratio for key/value tensors. Can be int or tuple (D,H,W)
            qkv_bias: Whether to add bias to qkv projections
            attn_dropout: Attention dropout rate
            proj_dropout: Projection dropout rate
            use_sdpa: Whether to use Scaled Dot-Product Attention
        """
        super().__init__()
        
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_sdpa = use_sdpa
        
        # QKV projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        
        # Spatial reduction for keys and values (if sr_ratio > 1)
        self.sr_ratio = sr_ratio
        if sr_ratio is not None:
            if isinstance(sr_ratio, int):
                sr_ratio = (sr_ratio, sr_ratio, sr_ratio)
            
            if any(r > 1 for r in sr_ratio):
                self.sr = nn.Conv3d(embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(embed_dim)
        
        # Attention and projection dropout
        if not use_sdpa:
            self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of self-attention.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Output tensor of same shape
        """
        B, C, D, H, W = x.shape
        
        # Flatten spatial dimensions for transformer processing
        x_flat = x.flatten(2).transpose(1, 2)  # (B, D*H*W, C)
        
        # Generate Q, K, V
        qkv = self.qkv(x_flat).reshape(B, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0)
        
        # Apply spatial reduction to keys and values if configured
        if hasattr(self, 'sr') and hasattr(self, 'norm'):
            # Reduce spatial resolution for K and V
            x_sr = self.sr(x)  # (B, C, D', H', W')
            _, _, D_sr, H_sr, W_sr = x_sr.shape
            x_sr_flat = x_sr.flatten(2).transpose(1, 2)  # (B, D'*H'*W', C)
            x_sr_flat = self.norm(x_sr_flat)
            
            # Generate reduced K, V
            kv = self.qkv(x_sr_flat).reshape(B, -1, 3, self.num_heads, self.head_dim)
            kv = kv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N', head_dim)
            _, k, v = kv.unbind(0)  # Use reduced k, v but keep original q
        
        # Compute attention using SDPA or manual implementation
        if self.use_sdpa and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's optimized SDPA
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.training * getattr(self, 'attn_dropout', nn.Dropout(0.0)).p if hasattr(self, 'attn_dropout') else 0.0,
                is_causal=False
            )
        else:
            # Manual attention computation
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            if hasattr(self, 'attn_dropout'):
                attn = self.attn_dropout(attn)
            attn_output = attn @ v
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(B, -1, C)
        attn_output = self.proj(attn_output)
        attn_output = self.proj_dropout(attn_output)
        
        # Reshape back to (B, C, D, H, W)
        attn_output = attn_output.transpose(1, 2).reshape(B, C, D, H, W)
        
        return attn_output


class DWConv(nn.Module):
    """Depth-wise convolution for 3D feature processing."""
    
    def __init__(self, embed_dim: int) -> None:
        """Initialize depth-wise convolution.
        
        Args:
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.dwconv = nn.Conv3d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, groups=embed_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Output tensor of same shape
        """
        return self.dwconv(x)


class Mlp(nn.Module):
    """Multi-layer perceptron with 3D convolution and optional depth-wise convolution."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0
    ) -> None:
        """Initialize MLP.
        
        Args:
            in_features: Input feature dimension
            hidden_features: Hidden feature dimension
            out_features: Output feature dimension
            act_layer: Activation layer class
            drop: Dropout rate
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Conv3d(in_features, hidden_features, kernel_size=1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Output tensor
        """
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with 3D self-attention and MLP."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        sr_ratio: int | tuple[int, int, int] | None = None,
        use_sdpa: bool = True
    ) -> None:
        """Initialize transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            qkv_bias: Whether to add bias to qkv projections
            drop: Dropout rate
            attn_drop: Attention dropout rate
            drop_path: Stochastic depth rate
            act_layer: Activation layer class
            norm_layer: Normalization layer class
            sr_ratio: Spatial reduction ratio for attention
            use_sdpa: Whether to use Scaled Dot-Product Attention
        """
        super().__init__()
        
        # Use custom layer normalization for 3D data
        self.norm1 = LayerNorm3d(embed_dim)
        self.attn = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            qkv_bias=qkv_bias,
            attn_dropout=attn_drop,
            proj_dropout=drop,
            use_sdpa=use_sdpa
        )
        
        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.norm2 = LayerNorm3d(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Output tensor of same shape
        """
        # Self-attention with residual connection
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        # MLP with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class LayerNorm3d(nn.Module):
    """Layer normalization for 3D tensors (B, C, D, H, W)."""
    
    def __init__(self, num_features: int, eps: float = 1e-6) -> None:
        """Initialize layer normalization.
        
        Args:
            num_features: Number of feature channels
            eps: Small value for numerical stability
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Normalized tensor
        """
        # Compute mean and variance across channel dimension
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, keepdim=True, unbiased=False)
        
        # Normalize
        x = (x - mean) / torch.sqrt(variance + self.eps)
        
        # Apply learnable parameters
        x = x * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1)
        
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    
    def __init__(self, drop_prob: float = 0.0) -> None:
        """Initialize drop path.
        
        Args:
            drop_prob: Drop probability
        """
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor (potentially dropped)
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # Work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        output = x.div(keep_prob) * random_tensor
        
        return output


class OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding for 3D input."""
    
    def __init__(
        self,
        img_size: tuple[int, int, int] = (224, 224, 224),
        patch_size: int | tuple[int, int, int] = 7,
        stride: int | tuple[int, int, int] = 4,
        in_chans: int = 3,
        embed_dim: int = 768
    ) -> None:
        """Initialize overlapping patch embedding.
        
        Args:
            img_size: Input image size (D, H, W)
            patch_size: Patch size
            stride: Patch stride
            in_chans: Number of input channels
            embed_dim: Embedding dimension
        """
        super().__init__()
        
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        
        # Calculate output size
        self.H = math.ceil((img_size[1] - patch_size[1]) / stride[1]) + 1
        self.W = math.ceil((img_size[2] - patch_size[2]) / stride[2]) + 1
        self.D = math.ceil((img_size[0] - patch_size[0]) / stride[0]) + 1
        
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2)
        )
        self.norm = LayerNorm3d(embed_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Embedded tensor of shape (B, embed_dim, D', H', W')
        """
        x = self.proj(x)
        x = self.norm(x)
        return x


class MixVisionTransformer(nn.Module):
    """Mix Vision Transformer encoder for SegFormer3D."""
    
    def __init__(
        self,
        img_size: tuple[int, int, int] = (224, 224, 224),
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dims: tuple[int, ...] = (64, 128, 256, 512),
        num_heads: tuple[int, ...] = (1, 2, 4, 8),
        mlp_ratios: tuple[float, ...] = (4, 4, 4, 4),
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[nn.Module] = LayerNorm3d,
        depths: tuple[int, ...] = (3, 4, 6, 3),
        sr_ratios: tuple[int, ...] = (8, 4, 2, 1),
        use_sdpa: bool = True
    ) -> None:
        """Initialize Mix Vision Transformer.
        
        Args:
            img_size: Input image size
            patch_size: Patch size for initial embedding
            in_chans: Number of input channels
            num_classes: Number of classes (not used in SegFormer)
            embed_dims: Embedding dimensions for each stage
            num_heads: Number of attention heads for each stage
            mlp_ratios: MLP expansion ratios for each stage
            qkv_bias: Whether to add bias to qkv projections
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            drop_path_rate: Stochastic depth rate
            norm_layer: Normalization layer class
            depths: Number of transformer blocks for each stage
            sr_ratios: Spatial reduction ratios for each stage
            use_sdpa: Whether to use Scaled Dot-Product Attention
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        # Patch embeddings for each stage
        self.patch_embeds = nn.ModuleList()
        self.pos_embeds = nn.ParameterList()
        self.pos_drops = nn.ModuleList()
        self.blocks:list[list[nn.Module]] = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(len(depths)):
            # Patch embedding
            if i == 0:
                patch_embed = OverlapPatchEmbed(
                    img_size=img_size,
                    patch_size=7 if patch_size == 4 else patch_size,
                    stride=4,
                    in_chans=in_chans,
                    embed_dim=embed_dims[i]
                )
            else:
                patch_embed = OverlapPatchEmbed(
                    img_size=(img_size[0] // (4 * 2**(i-1)), img_size[1] // (4 * 2**(i-1)), img_size[2] // (4 * 2**(i-1))),
                    patch_size=3,
                    stride=2,
                    in_chans=embed_dims[i-1],
                    embed_dim=embed_dims[i]
                )
            
            self.patch_embeds.append(patch_embed)
            
            # Position embedding (learnable)
            patch_resolution = (patch_embed.D, patch_embed.H, patch_embed.W)
            pos_embed = nn.Parameter(
                torch.zeros(1, embed_dims[i], patch_resolution[0], patch_resolution[1], patch_resolution[2])
            )
            self.pos_embeds.append(pos_embed)
            
            self.pos_drops.append(nn.Dropout(p=drop_rate))
            
            # Transformer blocks
            block_list = nn.ModuleList()
            for j in range(depths[i]):
                block = TransformerBlock(
                    embed_dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i]) + j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[i],
                    use_sdpa=use_sdpa
                )
                block_list.append(block)
            
            self.blocks.append(block_list)
            
            # Layer normalization
            self.norms.append(norm_layer(embed_dims[i]))
        
        # Initialize position embeddings
        for pos_embed in self.pos_embeds:
            nn.init.trunc_normal_(pos_embed, std=0.02)
    
    def _get_pos_embed(self, pos_embed: Tensor, patch_shape: tuple[int, int, int]) -> Tensor:
        """Interpolate position embedding to match patch shape.
        
        Args:
            pos_embed: Position embedding tensor
            patch_shape: Target patch shape (D, H, W)
            
        Returns:
            Interpolated position embedding
        """
        if patch_shape == pos_embed.shape[-3:]:
            return pos_embed
        
        return F.interpolate(
            pos_embed,
            size=patch_shape,
            mode='trilinear',
            align_corners=False
        )
    
    def forward_features(self, x: Tensor) -> list[Tensor]:
        """Forward pass through all stages.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            List of feature tensors from each stage
        """
        features = []
        
        for i in range(len(self.depths)):
            # Patch embedding
            x = self.patch_embeds[i](x)
            
            # Add position embedding
            pos_embed = self._get_pos_embed(self.pos_embeds[i], x.shape[-3:])
            x = x + pos_embed
            x = self.pos_drops[i](x)
            
            # Transformer blocks
            for block in self.blocks[i]:
                x = block(x)
            
            # Normalization
            x = self.norms[i](x)
            
            features.append(x)
        
        return features
    
    def forward(self, x: Tensor) -> list[Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            List of multi-scale features
        """
        return self.forward_features(x)


class SegFormerHead(nn.Module):
    """SegFormer decoder head for 3D segmentation."""
    
    def __init__(
        self,
        in_channels: tuple[int, ...],
        feature_strides: tuple[int, ...],
        embed_dim: int = 256,
        num_classes: int = 19,
        dropout_ratio: float = 0.1,
        align_corners: bool = False
    ) -> None:
        """Initialize SegFormer decoder head.
        
        Args:
            in_channels: Input channels for each feature level
            feature_strides: Feature strides for each level
            embed_dim: Embedding dimension for fusion
            num_classes: Number of segmentation classes
            dropout_ratio: Dropout ratio
            align_corners: Whether to align corners in upsampling
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.feature_strides = feature_strides
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.align_corners = align_corners
        
        # Linear layers to project features to same dimension
        self.linear_c = nn.ModuleList()
        for in_ch in in_channels:
            self.linear_c.append(
                nn.Sequential(
                    nn.Conv3d(in_ch, embed_dim, kernel_size=1),
                    LayerNorm3d(embed_dim)
                )
            )
        
        # Fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv3d(embed_dim * len(in_channels), embed_dim, kernel_size=1),
            LayerNorm3d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Classification head
        self.linear_pred = nn.Sequential(
            nn.Dropout3d(dropout_ratio),
            nn.Conv3d(embed_dim, num_classes, kernel_size=1)
        )
    
    def forward(self, features: list[Tensor]) -> Tensor:
        """Forward pass.
        
        Args:
            features: List of feature tensors from encoder
            
        Returns:
            Segmentation logits
        """
        # Get target size from first feature map
        target_size = features[0].shape[2:]
        
        # Project and upsample all features to same size
        projected_features = []
        for i, feature in enumerate(features):
            # Project to embed_dim
            proj_feat = self.linear_c[i](feature)
            
            # Upsample to target size
            if proj_feat.shape[2:] != target_size:
                proj_feat = F.interpolate(
                    proj_feat,
                    size=target_size,
                    mode='trilinear',
                    align_corners=self.align_corners
                )
            
            projected_features.append(proj_feat)
        
        # Concatenate and fuse features
        fused_features = torch.cat(projected_features, dim=1)
        fused_features = self.linear_fuse(fused_features)
        
        # Generate predictions
        logits = self.linear_pred(fused_features)
        
        return logits


class SegFormer3D(nn.Module):
    """SegFormer3D model for 3D medical image segmentation.
    
    This is a standalone PyTorch model that can be used with any training framework.
    """
    
    def __init__(
        self,
        img_size: Sequence[int],
        in_chans: int = 1,
        num_classes: int = 1,
        embed_dims: tuple[int, ...] = (64, 128, 256, 512),
        num_heads: tuple[int, ...] = (1, 2, 4, 8),
        mlp_ratios: tuple[float, ...] = (2, 2, 2, 2),
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        depths: tuple[int, ...] = (2, 2, 2, 2),
        sr_ratios: tuple[int, ...] = (8, 4, 2, 1),
        decoder_embed_dim: int = 256,
        decoder_dropout: float = 0.1,
        use_sdpa: bool = True,
        align_corners: bool = False
    ) -> None:
        """Initialize SegFormer3D model.
        
        Args:
            img_size: Input image size (D, H, W)
            in_chans: Number of input channels
            num_classes: Number of segmentation classes
            embed_dims: Embedding dimensions for each encoder stage
            num_heads: Number of attention heads for each stage
            mlp_ratios: MLP expansion ratios for each stage
            qkv_bias: Whether to add bias to qkv projections
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            drop_path_rate: Stochastic depth rate
            depths: Number of transformer blocks for each stage
            sr_ratios: Spatial reduction ratios for each stage
            decoder_embed_dim: Decoder embedding dimension
            decoder_dropout: Decoder dropout rate
            use_sdpa: Whether to use Scaled Dot-Product Attention
            align_corners: Whether to align corners in upsampling
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Encoder
        self.encoder = MixVisionTransformer(
            img_size=img_size,
            patch_size=4,
            in_chans=in_chans,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            depths=depths,
            sr_ratios=sr_ratios,
            use_sdpa=use_sdpa
        )
        
        # Decoder
        feature_strides = tuple(4 * 2**i for i in range(len(embed_dims)))
        self.decoder = SegFormerHead(
            in_channels=embed_dims,
            feature_strides=feature_strides,
            embed_dim=decoder_embed_dim,
            num_classes=num_classes,
            dropout_ratio=decoder_dropout,
            align_corners=align_corners
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Segmentation logits of shape (B, num_classes, D, H, W)
        """
        # Encoder
        features = self.encoder(x)
        
        # Decoder
        logits = self.decoder(features)
        
        # Upsample to input size if needed
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(
                logits,
                size=x.shape[2:],
                mode='trilinear',
                align_corners=self.decoder.align_corners
            )
        
        return logits
