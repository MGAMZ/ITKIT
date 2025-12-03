"""
@InProceedings{Perera_2024_CVPR,
    author    = {Perera, Shehan and Navard, Pouyan and Yilmaz, Alper},
    title     = {SegFormer3D: An Efficient Transformer for 3D Medical Image Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {4981-4988}
}

Modified to support arbitrary DWH input and optimized using SDPA.
"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        sr_ratio=None,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        use_SDPA: bool = True,
    ):
        super().__init__()
        assert (embed_dim % num_heads == 0), \
            "Embedding dim must be divisible by number of heads!"

        self.num_heads = num_heads
        self.attention_head_dim = embed_dim // num_heads
        self.scale:float = self.attention_head_dim ** -0.5

        self.query = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.key_value = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.attn_dropout_p = attn_dropout
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.use_SDPA = use_SDPA and hasattr(F, "scaled_dot_product_attention")

        if not self.use_SDPA:
            print("Warning: scaled_dot_product_attention not available or disabled. Using manual attention implementation.")

        if sr_ratio is None:
            sr_ratio = 1
        if isinstance(sr_ratio, int):
            self.sr_ratio = (sr_ratio, sr_ratio, sr_ratio)
        else:
            assert len(sr_ratio) == 3
            self.sr_ratio = tuple(sr_ratio)
        if any(r > 1 for r in self.sr_ratio):
            self.sr = nn.Conv3d(embed_dim, embed_dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            self.sr_norm = nn.LayerNorm(embed_dim)
        else:
            self.sr = None

    def forward(self, x:torch.Tensor, patched_volume_size):
        B, N, C = x.shape
        D, W, H = patched_volume_size

        # q shape: (B, num_heads, N, head_dim)
        q:torch.Tensor = self.query(x).view(B, N, self.num_heads, self.attention_head_dim).permute(0, 2, 1, 3)

        if self.sr is not None:
            # x shape: (B, N, C) -> (B, C, N) -> (B, C, D, W, H)
            x_ = x.permute(0, 2, 1).view(B, C, D, W, H)
            x_ = self.sr(x_)
            # x_ shape: (B, C, D'*W'*H') -> (B, D'*W'*H', C)
            x_ = x_.flatten(2).transpose(1, 2)
            x_ = self.sr_norm(x_)
            # kv shape: (2, B, num_heads, N_kv, head_dim) where N_kv = D'*W'*H'
            kv = self.key_value(x_)
            kv = kv.view(B, -1, 2, self.num_heads, self.attention_head_dim).permute(2, 0, 3, 1, 4)
        else:
            # kv shape: (2, B, num_heads, N, head_dim)
            kv = self.key_value(x)
            kv = kv.view(B, N, 2, self.num_heads, self.attention_head_dim).permute(2, 0, 3, 1, 4)

        # k, v shape: (B, num_heads, N_kv, head_dim)
        k, v = kv.unbind(0)

        if self.use_SDPA:
            # attn_output shape: (B, num_heads, N, head_dim)
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
            )
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_dropout(attn)
            # attn_output shape: (B, num_heads, N, head_dim)
            attn_output = attn @ v

        # attn_output shape: (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim) -> (B, N, C)
        out = attn_output.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out


class DWConv(nn.Module):
    """Depthwise Separable Convolution used in MLP"""
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm3d(dim) # Consider replacing with LayerNorm if issues arise

    def forward(self, x, patched_volume_size):
        B, N, C = x.shape
        # Use the provided spatial dimensions
        D, W, H = patched_volume_size
        if N == 0: # Handle empty sequences if they occur
            return x

        # Reshape for convolution: (B, N, C) -> (B, C, N) -> (B, C, D, W, H)
        x = x.transpose(1, 2).view(B, C, D, W, H)
        x = self.dwconv(x)
        x = self.bn(x)
        # Reshape back to sequence: (B, C, D, W, H) -> (B, C, N) -> (B, N, C)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlockMLP(nn.Module):
    """MLP block with Depthwise Separable Convolution"""
    def __init__(self, in_features: int, mlp_ratio: int, dropout: float):
        super().__init__()
        hidden_features = mlp_ratio * in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(dim=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, patched_volume_size):
        x = self.fc1(x)
        # Pass spatial dimensions to DWConv
        x = self.dwconv(x, patched_volume_size)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int = 4, # Default MLP ratio often 4
        sr_ratio: int = 1,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0, # Typically same dropout for attn proj and mlp
        use_SDPA: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            use_SDPA=use_SDPA,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = TransformerBlockMLP(
            in_features=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=proj_dropout, # Use proj_dropout for MLP dropout
        )

    def forward(self, x, patched_volume_size):
        # Pass spatial dimensions to attention and MLP
        x = x + self.attn(self.norm1(x), patched_volume_size)
        x = x + self.mlp(self.norm2(x), patched_volume_size)
        return x


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__()
        # Convolutional layer for patch embedding
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        # Layer normalization applied after flattening
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Apply convolution: (B, C_in, D, W, H) -> (B, C_embed, D', W', H')
        x = self.proj(x)
        # Get the spatial dimensions after convolution
        patched_volume_size = x.shape[2:]
        # Flatten spatial dimensions and transpose: (B, C_embed, D'*W'*H') -> (B, D'*W'*H', C_embed)
        x = x.flatten(2).transpose(1, 2)
        # Apply layer normalization
        x = self.norm(x)
        # Return both the embedded sequence and the spatial dimensions
        return x, patched_volume_size


class MixVisionTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dims: Sequence,
        num_heads: Sequence,
        mlp_ratios: Sequence,
        depths: Sequence,
        sr_ratios: Sequence,
        patch_kernel_size: Sequence,
        patch_stride: Sequence,
        patch_padding: Sequence,
        qkv_bias: bool = True, # Common default
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        use_SDPA: bool = True,
    ):
        super().__init__()
        self.depths = depths
        self.embed_dims = embed_dims

        # Create Patch Embedding layers for each stage
        self.patch_embeds = nn.ModuleList()
        # Input channels for the first stage is in_channels, subsequent stages use previous embed_dim
        input_ch = in_channels
        for i in range(len(depths)):
            self.patch_embeds.append(
                PatchEmbedding(
                    in_channels=input_ch,
                    embed_dim=embed_dims[i],
                    kernel_size=patch_kernel_size[i],
                    stride=patch_stride[i],
                    padding=patch_padding[i],
                )
            )
            input_ch = embed_dims[i] # Update input channel for the next stage

        # Create Transformer Blocks for each stage
        self.blocks = nn.ModuleList()
        for i in range(len(depths)):
            stage_blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        embed_dim=embed_dims[i],
                        num_heads=num_heads[i],
                        mlp_ratio=mlp_ratios[i],
                        sr_ratio=sr_ratios[i],
                        qkv_bias=qkv_bias,
                        attn_dropout=attn_dropout,
                        proj_dropout=proj_dropout,
                        use_SDPA=use_SDPA,
                    )
                    for _ in range(depths[i])
                ]
            )
            self.blocks.append(stage_blocks)

        # Layer Normalization after each stage's blocks
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dims[i]) for i in range(len(depths))])

    def forward(self, x):
        outputs = []
        B = x.shape[0]

        for i in range(len(self.depths)):
            # Apply patch embedding and get spatial size
            x, patched_volume_size = self.patch_embeds[i](x)
            D, W, H = patched_volume_size

            # Apply transformer blocks for the current stage
            for blk in self.blocks[i]: # type:ignore
                # Pass the spatial size to each block
                x = blk(x, patched_volume_size)

            # Apply normalization
            x = self.norms[i](x)

            # Reshape back to spatial format (B, C, D, W, H) for output
            C = x.shape[-1]
            x = x.transpose(1, 2).view(B, C, D, W, H)
            outputs.append(x)
            # The output 'x' becomes the input for the next stage's PatchEmbedding

        return outputs


class SegFormerDecoderHead(nn.Module):
    def __init__(
        self,
        input_feature_dims: list, # Embed dims from encoder stages [C1, C2, C3, C4]
        decoder_head_embedding_dim: int,
        final_upsampler_scale_factor: int|tuple[int],
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        class DecoderMapping(nn.Module):
            """Conv Embedding for Decoder"""
            def __init__(self, input_dim: int, embed_dim: int):
                super().__init__()
                # Use 1x1x1 Conv to project channels, acts on Volume format
                self.proj = nn.Conv3d(input_dim, embed_dim, kernel_size=1, stride=1, padding=0)
                # Use BatchNorm or GroupNorm for Volume format
                self.norm = nn.BatchNorm3d(embed_dim)
                # Or: self.norm = nn.GroupNorm(num_groups=..., num_channels=embed_dim)

            def forward(self, x):
                # Input x: (B, C_in, D, W, H)
                x = self.proj(x)
                x = self.norm(x)
                # Output x: (B, C_embed, D, W, H)
                return x

        # Conv embedding layers for features from each encoder stage
        self.mlps = nn.ModuleList() # Keep name mlps for consistency or rename
        for i in range(len(input_feature_dims)):
            self.mlps.append(
                # Use the new Conv-based mapping layer
                DecoderMapping(
                    input_dim=input_feature_dims[i],
                    embed_dim=decoder_head_embedding_dim,
                )
            )

        # ... (rest of __init__ remains the same) ...
        self.fuse = nn.Sequential(
            nn.Conv3d(
                in_channels=decoder_head_embedding_dim * len(input_feature_dims),
                out_channels=decoder_head_embedding_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm3d(decoder_head_embedding_dim),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.predict = nn.Conv3d(decoder_head_embedding_dim, num_classes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=final_upsampler_scale_factor,
                                    mode="trilinear", align_corners=False)

    def forward(self, encoder_features):
        # encoder_features is a list [c1, c2, c3, c4] from MixVisionTransformer
        B = encoder_features[0].shape[0]
        target_size = encoder_features[0].shape[2:] # Spatial size of the first stage

        all_features = []
        for i in range(len(encoder_features)):
            # Apply Conv mapping directly on the Volume feature map
            feat = self.mlps[i](encoder_features[i]) # Output: (B, C_decoder, Di, Wi, Hi)

            # Interpolate to the target size (size of c1) if not already there
            if feat.shape[2:] != target_size:
                feat = F.interpolate(
                    feat,
                    size=target_size,
                    mode="trilinear",
                    align_corners=False,
                )
            all_features.append(feat)

        # Concatenate features along the channel dimension
        fused_features = torch.cat(all_features, dim=1)
        fused_features = self.fuse(fused_features)

        out = self.dropout(fused_features)
        out = self.predict(out)
        out = self.upsample(out)

        return out


class SegFormer3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        # Default parameters inspired by SegFormer B0/B1 adapted for 3D
        embed_dims: list = [32, 64, 160, 256],
        num_heads: list = [1, 2, 5, 8],
        mlp_ratios: list = [4, 4, 4, 4],
        depths: list = [2, 2, 2, 2],
        sr_ratios: list[int|tuple[int,int,int]] = [4, 2, 1, 1], # Sequence reduction ratios per stage
        patch_kernel_size: list[int|tuple[int,int,int]] = [7, 3, 3, 3],
        patch_stride: list[int|tuple[int,int,int]] = [4, 2, 2, 2],
        patch_padding: list[int|tuple[int,int,int]] = [3, 1, 1, 1],
        decoder_head_embedding_dim: int = 256,
        num_classes: int = 3,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        decoder_dropout: float = 0.0,
        use_SDPA: bool = True,
    ):
        super().__init__()

        self.encoder = MixVisionTransformer(
            in_channels=in_channels,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            depths=depths,
            sr_ratios=sr_ratios,
            patch_kernel_size=patch_kernel_size,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            use_SDPA=use_SDPA,
        )

        self.decoder = SegFormerDecoderHead(
            # Decoder receives features in the order they are output by encoder
            input_feature_dims=embed_dims,
            decoder_head_embedding_dim=decoder_head_embedding_dim,
            final_upsampler_scale_factor=patch_stride[0],
            num_classes=num_classes,
            dropout=decoder_dropout,
        )

        # 注入 Grad-CAM 用的属性和 hook
        self.feature_maps = None
        self.gradients = None

    def forward(self, x):
        encoder_features = self.encoder(x) # [N, C, Z, Y, X]
        segmentation_output = self.decoder(encoder_features)
        return segmentation_output # [N, C, Z, Y, X]
