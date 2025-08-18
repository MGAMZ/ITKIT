"""Volume-level VMamba extension.

Components:
1. Slice feature extraction interface (wrapper around Backbone_VSSM)
2. VolumeVSSM class to process 3D volumes (B, C, D, H, W) -> volume embedding
3. 1D Mamba aggregator over slice sequence (linear complexity in depth D)
4. Assumption: batch size = 1 for simplicity (can be extended later)
5. Multi-task head producing risk/task outputs

This file is intentionally light-touch and does not modify vmamba.py.
"""

import pdb, math

import torch
import torch.nn as nn
import torch.nn.functional as F

from selective_scan import selective_scan_fn
import torch.utils.checkpoint
from vmamba import mamba_init # pyright: ignore[reportMissingImports]


class MambaAggregator1D(nn.Module, mamba_init):
    """Linear-time selective scan over slice embeddings (Depth dimension).

    Input: (B, D, C_in)
    Output: (B, C_out)

    Simplified from SS2Dv0 (single direction, no cross scanning).
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        dt_rank: int | str = "auto",
        dropout: float = 0.0,
        out_dim: int | None = None,
        use_conv: bool = True,
        pool: str = "mean",  # 'mean' | 'last'
        act_layer: nn.Module = nn.SiLU,
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.pool = pool
        d_inner = int(ssm_ratio * d_model)
        if dt_rank == "auto":
            dt_rank_val = math.ceil(d_model / 16)
        else:
            dt_rank_val = int(dt_rank)
        self.d_inner = d_inner
        # resolved integer rank
        self.dt_rank = dt_rank_val
        self.out_dim = d_model if out_dim is None else out_dim

        # in projection & activation
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.act = act_layer()

        # optional depthwise conv over sequence length (implemented as 1D conv)
        self.use_conv = use_conv
        if use_conv:
            self.dwconv = nn.Conv1d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=True)

        # parameter projections for SSM
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
        self.A_logs = self.A_log_init(d_state, d_inner, copies=1, merge=True)
        self.Ds = self.D_init(d_inner, copies=1, merge=True)

        # output layers
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, self.out_dim, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, "Expected (B, D, C)"
        B, D, _ = x.shape
        x = self.in_proj(x)  # (B,D,2*d_inner)
        x, z = x.split(self.d_inner, dim=-1)
        z = self.act(z)
        x = x.transpose(1, 2)  # (B,d_inner,D)
        if self.use_conv:
            x = self.dwconv(x)
        x = self.act(x)
        x_seq = x.transpose(1, 2)  # (B,D,d_inner)
        x_dbl = self.x_proj(x_seq)  # (B,D,dt_rank+2*d_state)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dts = self.dt_proj(dts)  # (B,D,d_inner)

        u = x  # (B,d_inner,D)
        delta = dts.transpose(1, 2)
        Bs = Bs.transpose(1, 2).unsqueeze(1)  # (B,1,d_state,D)
        Cs = Cs.transpose(1, 2).unsqueeze(1)
        As = -torch.exp(self.A_logs.float())  # (d_inner,d_state)
        Ds = self.Ds.float()
        dt_bias = self.dt_proj.bias.float()

        y = selective_scan_fn(
            u,          # u: (B, d_inner, D)
            delta,      # delta: (B, d_inner, D)
            As,         # A: (d_inner, d_state)
            Bs,         # B: (B, 1, d_state, D)
            Cs,         # C: (B, 1, d_state, D)
            Ds,         # D: (d_inner,)
            delta_bias=dt_bias,
            delta_softplus=True,
        )  # (B, d_inner, D)
        y = y.transpose(1, 2)  # (B, D, d_inner)
        y = self.out_norm(y)
        y = y * z

        if self.pool == "mean":
            v = y.mean(dim=1)
        elif self.pool == "last":
            v = y[:, -1]
        else:
            raise ValueError(f"Unsupported pool mode: {self.pool}")

        v = self.dropout(self.out_proj(v))
        return v


class SliceFeatureExtractor(nn.Module):
    def __init__(self, backbone: torch.nn.Module, pool: str = "gap", use_torch_ckpt:bool=False):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.use_torch_ckpt = use_torch_ckpt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Z, Y, X = x.shape
        slices = x.permute(0, 2, 1, 3, 4).reshape(B * Z, C, Y, X) # [N, C, Z, Y, X] -> [N*Z, C, Y, X]
        
        if self.use_torch_ckpt:
            feats = torch.utils.checkpoint.checkpoint(self.backbone, slices, use_reentrant=False)  # list of [N*Z, latent, Y, X]
        else:
            feats = self.backbone(slices)  # list of [N*Z, latent, Y, X]
        if isinstance(feats, (list, tuple)):
            f = feats[-1]
        else:
            f = feats
        
        _, C, Y, X = f.shape
        
        f = f.reshape(B, Z, C, Y, X).permute(0, 2, 1, 3, 4)  # [N*Z, C, Y, X] -> [N, C, Z, Y, X]
        if self.pool == 'gap':
            f = f.mean(dim=[-1, -2])
        else:
            raise ValueError(f"Unsupported pool {self.pool}")
        
        return f # [N, C, Z]


class VolumeVSSM(nn.Module):
    """3D volume -> slice encoder (2D VMamba) -> depth aggregator (1D Mamba) -> multi-task heads.

    Args:
        backbone_kwargs: kwargs for Backbone_VSSM
        aggregator_kwargs: kwargs for MambaAggregator1D (d_model derived if absent)
        task_out_dims: dict or int
        slice_pool: pooling over spatial dims per slice ('gap' or 'max')

    Input shape: (B,C,D,H,W); assuming B=1 currently.
    """
    def __init__(
        self,
        slice_extractor_backbone: torch.nn.Module,
        aggregator: MambaAggregator1D,
    ):
        super().__init__()
        self.slice_extractor = SliceFeatureExtractor(slice_extractor_backbone)
        self.embed_dims = self.slice_extractor.backbone.dims
        self.aggregator = aggregator

    def forward(self, vol: torch.Tensor):
        assert vol.dim() == 5, 'Volume must be (B,C,D,H,W)'
        B, C, D, H, W = vol.shape
        
        slice_emb = self.slice_extractor(vol)  # return: [B, C, Z]
        vol_emb = self.aggregator(slice_emb.permute(0,2,1))  # (B, C_out)
        
        return (vol_emb, )


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VolumeVSSM(
        backbone_kwargs=dict(in_chans=1, depths=[1,1,2,1], dims=[32,64,128,256], norm_layer='ln'),
        aggregator_kwargs=dict(d_model=256, ssm_ratio=2.0, d_state=16, pool='mean'),
    ).to(device)
    
    with torch.no_grad():
        vol = torch.randn(1,1,12,128,128, device=device) # [N, C, Z, Y, X]
        outputs = model(vol)
        print('Embedding shape:', outputs.shape)
