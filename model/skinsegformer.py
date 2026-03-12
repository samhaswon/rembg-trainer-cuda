from dataclasses import dataclass
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.utils.flop_counter import FlopCounterMode


def _pair(x: int) -> Tuple[int, int]:
    return (x, x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiheadSelfAttention(nn.Module):
    """
    Self-attention using PyTorch scaled_dot_product_attention when available.
    """
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0, force_explicit: bool = False) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.force_explicit = force_explicit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n, dim = x.shape
        qkv = self.qkv(x)  # (B, N, 3D)
        qkv = qkv.view(bsz, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, Hd)

        # Use SDPA if available (PyTorch 2.x). This can dispatch to FlashAttention on supported GPUs.
        if (not self.force_explicit) and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_drop if self.training else 0.0,
                is_causal=False,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            if self.attn_drop > 0:
                attn = nn.functional.dropout(attn, p=self.attn_drop, training=self.training)
            out = attn @ v

        out = out.transpose(1, 2).contiguous().view(bsz, n, dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MultiheadCrossAttention(nn.Module):
    """
    Cross-attention: queries from x, keys/values from ctx.
    """
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        bsz, n, dim = x.shape
        m = ctx.shape[1]

        q = self.q(x).view(bsz, n, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,N,Hd)
        kv = self.kv(ctx).view(bsz, m, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (B,H,M,Hd)

        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_drop if self.training else 0.0,
                is_causal=False,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            if self.attn_drop > 0:
                attn = nn.functional.dropout(attn, p=self.attn_drop, training=self.training)
            out = attn @ v

        out = out.transpose(1, 2).contiguous().view(bsz, n, dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        use_rmsnorm: bool = True,
    ) -> None:
        super().__init__()
        norm = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.norm1 = norm(dim)
        self.attn = MultiheadSelfAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class BiRefineDecoderBlock(nn.Module):
    """
    BiRefNet-inspired refinement without deformable convs:
    - global token stream (xg) attends to detail stream (xd)
    - detail stream (xd) attends back to global stream (xg)
    - both have local self-attn + MLP
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        use_rmsnorm: bool = True,
    ) -> None:
        super().__init__()
        norm = RMSNorm if use_rmsnorm else nn.LayerNorm

        self.g_norm1 = norm(dim)
        self.g_self = MultiheadSelfAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.g_norm2 = norm(dim)
        self.g_xattn = MultiheadCrossAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.g_norm3 = norm(dim)
        self.g_mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

        self.d_norm1 = norm(dim)
        self.d_self = MultiheadSelfAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.d_norm2 = norm(dim)
        self.d_xattn = MultiheadCrossAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.d_norm3 = norm(dim)
        self.d_mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, xg: torch.Tensor, xd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xg = xg + self.g_self(self.g_norm1(xg))
        xg = xg + self.g_xattn(self.g_norm2(xg), ctx=xd)
        xg = xg + self.g_mlp(self.g_norm3(xg))

        xd = xd + self.d_self(self.d_norm1(xd))
        xd = xd + self.d_xattn(self.d_norm2(xd), ctx=xg)
        xd = xd + self.d_mlp(self.d_norm3(xd))
        return xg, xd


class TopKMoEMLP(nn.Module):
    """
    Top-k routed MoE MLP:
    - increases parameters with more experts
    - keeps token compute controlled by limiting experts per token
    """
    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int = 2,
        mlp_ratio: float = 1.75,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1.")
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.gate = nn.Linear(dim, num_experts, bias=True)
        self.experts = nn.ModuleList([
            MLP(dim, mlp_ratio=mlp_ratio, drop=drop)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n, dim = x.shape
        gate_logits = self.gate(x)  # (B, N, E)
        topk_logits, topk_idx = torch.topk(gate_logits, k=self.top_k, dim=-1)  # (B,N,K), (B,N,K)
        topk_weight = topk_logits.softmax(dim=-1)  # normalize over selected experts only

        x_flat = x.reshape(-1, dim)
        out_flat = torch.zeros_like(x_flat)
        idx_flat = topk_idx.reshape(-1, self.top_k)
        weight_flat = topk_weight.reshape(-1, self.top_k)

        for expert_id, expert in enumerate(self.experts):
            mask = idx_flat == expert_id  # (T, K)
            if mask.any():
                token_ids, slot_ids = mask.nonzero(as_tuple=True)
                expert_out = expert(x_flat[token_ids])
                expert_weight = weight_flat[token_ids, slot_ids].unsqueeze(-1)
                out_flat.index_add_(0, token_ids, expert_out * expert_weight)

        return out_flat.view(bsz, n, dim)


class BiRefineMoEDecoderBlock(nn.Module):
    """
    BiRefine block with MoE MLPs in both streams.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_experts: int,
        moe_top_k: int = 2,
        moe_mlp_ratio: float = 1.75,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        use_rmsnorm: bool = True,
    ) -> None:
        super().__init__()
        norm = RMSNorm if use_rmsnorm else nn.LayerNorm

        self.g_norm1 = norm(dim)
        self.g_self = MultiheadSelfAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.g_norm2 = norm(dim)
        self.g_xattn = MultiheadCrossAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.g_norm3 = norm(dim)
        self.g_moe = TopKMoEMLP(
            dim,
            num_experts=num_experts,
            top_k=moe_top_k,
            mlp_ratio=moe_mlp_ratio,
            drop=drop,
        )

        self.d_norm1 = norm(dim)
        self.d_self = MultiheadSelfAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.d_norm2 = norm(dim)
        self.d_xattn = MultiheadCrossAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.d_norm3 = norm(dim)
        self.d_moe = TopKMoEMLP(
            dim,
            num_experts=num_experts,
            top_k=moe_top_k,
            mlp_ratio=moe_mlp_ratio,
            drop=drop,
        )

    def forward(self, xg: torch.Tensor, xd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xg = xg + self.g_self(self.g_norm1(xg))
        xg = xg + self.g_xattn(self.g_norm2(xg), ctx=xd)
        xg = xg + self.g_moe(self.g_norm3(xg))

        xd = xd + self.d_self(self.d_norm1(xd))
        xd = xd + self.d_xattn(self.d_norm2(xd), ctx=xg)
        xd = xd + self.d_moe(self.d_norm3(xd))
        return xg, xd


class PatchEmbed(nn.Module):
    """
    Patchify a 512x512 canvas into (H/P)*(W/P) tokens.
    """
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, C, H/P, W/P)
        bsz, dim, hp, wp = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x


class InterpCanvasProjector(nn.Module):
    """
    Resize to a 512x512 canvas, then a small trainable stem at 512.

    Returns:
      - x_512: (B, stem_ch, 512, 512) for ViT patch embedding
      - feat_mid: (B, stem_ch, H/2, W/2) aligned with the original input resolution
    """
    def __init__(self, in_chans: int, stem_ch: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, stem_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.GELU(),
            nn.Conv2d(stem_ch, stem_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, ch, h, w = x.shape
        if not (h == w == 512 or h == w == 1024):
            raise ValueError(f"Only 512x512 or 1024x1024 supported, got {h}x{w}.")

        # Intermediates: allowed to resize non-trainably
        x_512_img = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)

        x_512 = self.stem(x_512_img)  # (B, stem_ch, 512, 512)

        # Need feat_mid at (H/2, W/2)
        mid_hw = (h // 2, w // 2)
        if mid_hw == (512, 512):
            # Input was 1024, mid-res is 512: upsample stem output to mid-res
            feat_mid = x_512
        else:
            # Input was 512, mid-res is 256: downsample stem output
            feat_mid = F.interpolate(x_512, size=mid_hw, mode="bilinear", align_corners=False)

        return x_512, feat_mid


class MidResRefineHead(nn.Module):
    """
    Token logits are refined at H/2 using aligned conv features, then a trainable final upsample produces H×W logits.
    """
    def __init__(
        self,
        dec_dim: int,
        num_classes: int,
        patch_grid: int = 32,
        mid_feat_ch: int = 64,
        fuse_ch: int = 128,
        refine_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.patch_grid = patch_grid

        self.tok_proj = nn.Linear(dec_dim, fuse_ch, bias=True)
        self.mid_proj = nn.Sequential(
            nn.Conv2d(mid_feat_ch, fuse_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(fuse_ch),
            nn.GELU(),
        )

        blocks = []
        for _ in range(refine_blocks):
            blocks += [
                nn.Conv2d(fuse_ch, fuse_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fuse_ch),
                nn.GELU(),
            ]
        self.refine = nn.Sequential(*blocks)

        # Final trainable resize: (H/2,W/2)->(H,W)
        self.final_up2 = nn.Sequential(
            nn.ConvTranspose2d(fuse_ch, fuse_ch, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(fuse_ch),
            nn.GELU(),
        )
        self.out = nn.Conv2d(fuse_ch, num_classes, kernel_size=1, bias=True)

    def forward(self, x_tokens: torch.Tensor, feat_mid: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        bsz, n, _ = x_tokens.shape
        grid = self.patch_grid
        if n != grid * grid:
            raise ValueError(f"Expected {grid*grid} tokens, got {n}.")

        h, w = out_hw
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(f"Expected even H,W for final 2x upsample, got {h}x{w}.")

        # Tokens -> 32x32 feature map
        x = self.tok_proj(x_tokens).transpose(1, 2).contiguous().view(bsz, -1, grid, grid)

        # Resize tokens to mid-res (allowed to be non-trainable)
        x = nn.functional.interpolate(
            x, size=(h // 2, w // 2), mode="nearest"
        )

        # Fuse aligned mid-res conv detail
        x = x + self.mid_proj(feat_mid)

        # Refine at mid-res
        x = self.refine(x)

        # Final trainable resize to full res and output logits
        x = self.final_up2(x)
        return self.out(x)


class TokenToLogitsFinalTrainableUpsample(nn.Module):
    """
    Intermediates may be resized with interpolate.
    Final step to exact output resolution is trainable (ConvTranspose2d),
    so logits are produced at input resolution without non-trainable resizing.
    """
    def __init__(
        self,
        dec_dim: int,
        num_classes: int,
        patch_grid: int = 32,
        mid_ch: int = 96,
    ) -> None:
        super().__init__()
        self.patch_grid = patch_grid
        self.proj = nn.Linear(dec_dim, mid_ch, bias=True)

        # Trainable final 2x upsample: (H/2,W/2) -> (H,W)
        self.final_up2 = nn.Sequential(
            nn.ConvTranspose2d(mid_ch, mid_ch, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
        )
        self.out = nn.Conv2d(mid_ch, num_classes, kernel_size=1, bias=True)

    def forward(self, x_tokens: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        bsz, n, _ = x_tokens.shape
        grid = self.patch_grid
        if n != grid * grid:
            raise ValueError(f"Expected {grid*grid} tokens, got {n}.")

        h, w = out_hw
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(f"Expected even H,W for final 2x upsample, got {h}x{w}.")

        x = self.proj(x_tokens)  # (B, N, mid_ch)
        x = x.transpose(1, 2).contiguous().view(bsz, -1, grid, grid)  # (B, mid_ch, 32, 32)

        # Non-trainable intermediate resize to half resolution
        x = F.interpolate(x, size=(h // 2, w // 2), mode="bilinear", align_corners=False)

        # Trainable final resize to full resolution
        x = self.final_up2(x)
        logits = self.out(x)
        return logits


class ViTEncoder(nn.Module):
    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        patch_size: int,
        mlp_ratio: float,
        drop: float,
        attn_drop: float,
        use_abs_pos: bool = True,
        use_rmsnorm: bool = True,
        grad_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.patch = PatchEmbed(in_chans, embed_dim, patch_size=patch_size)
        self.use_abs_pos = use_abs_pos
        self.grad_checkpointing = grad_checkpointing

        # 512 / patch_size tokens per side
        tokens_side = 512 // patch_size
        n_tokens = tokens_side * tokens_side

        if use_abs_pos:
            self.pos = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
            nn.init.trunc_normal_(self.pos, std=0.02)
        else:
            self.pos = None

        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                use_rmsnorm=use_rmsnorm,
            )
            for _ in range(depth)
        ])
        self.norm = (RMSNorm(embed_dim) if use_rmsnorm else nn.LayerNorm(embed_dim))

    def forward(self, x_512: torch.Tensor) -> torch.Tensor:
        x = self.patch(x_512)  # (B, N, D)
        if self.pos is not None:
            x = x + self.pos
        x = self.drop(x)

        for blk in self.blocks:
            if self.grad_checkpointing and self.training:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.norm(x)
        return x  # (B, N, D)


class DetailTokenExtractor(nn.Module):
    """
    Extract "detail tokens" from high-res features, but match the same token grid as the ViT.
    No interpolation: use strided convs to reach 32x32 when input is 1024, and 32x32 when input is 512.
    """
    def __init__(self, in_chans: int, out_dim: int, patch_size: int) -> None:
        super().__init__()
        # Target token grid = 512/patch_size = 32 for patch_size=16.
        self.patch_size = patch_size
        self.to_tokens_512 = nn.Sequential(
            nn.Conv2d(in_chans, out_dim, kernel_size=3, stride=2, padding=1, bias=False),  # 512->256
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),  # 256->128
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),  # 128->64
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),  # 64->32
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )
        self.to_tokens_1024 = nn.Sequential(
            nn.Conv2d(in_chans, out_dim, kernel_size=3, stride=2, padding=1, bias=False),  # 1024->512
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),  # 512->256
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),  # 256->128
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),  # 128->64
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),  # 64->32
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

    def forward(self, feat_hi: torch.Tensor) -> torch.Tensor:
        h, w = feat_hi.shape[-2:]
        if (h, w) == (512, 512):
            x = self.to_tokens_512(feat_hi)
        elif (h, w) == (1024, 1024):
            x = self.to_tokens_1024(feat_hi)
        else:
            raise ValueError(f"Only 512x512 or 1024x1024 supported, got {h}x{w}.")

        bsz, dim, ht, wt = x.shape  # ht=wt=32
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        enc_dim: int,
        dec_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        drop: float,
        attn_drop: float,
        moe_depth: int,
        num_experts: int,
        moe_top_k: int = 2,
        moe_mlp_ratio: float = 1.75,
        use_rmsnorm: bool = True,
        grad_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if moe_depth < 0:
            raise ValueError("moe_depth must be >= 0.")
        self.grad_checkpointing = grad_checkpointing
        self.enc_to_dec = nn.Linear(enc_dim, dec_dim, bias=True)

        # Keep MoE blocks wrapped by dense blocks: at least one dense at input and output.
        max_sparse_depth = max(0, depth - 2)
        sparse_depth = min(moe_depth, max_sparse_depth)
        leading_dense = 1 if sparse_depth > 0 else 0
        trailing_dense = 1 if sparse_depth > 0 else 0
        middle_dense = depth - sparse_depth - leading_dense - trailing_dense

        blocks = []
        blocks.extend([
            BiRefineDecoderBlock(
                dim=dec_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                use_rmsnorm=use_rmsnorm,
            )
            for _ in range(leading_dense)
        ])
        blocks.extend([
            BiRefineMoEDecoderBlock(
                dim=dec_dim,
                num_heads=num_heads,
                num_experts=num_experts,
                moe_top_k=moe_top_k,
                moe_mlp_ratio=moe_mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                use_rmsnorm=use_rmsnorm,
            )
            for _ in range(sparse_depth)
        ])
        blocks.extend([
            BiRefineDecoderBlock(
                dim=dec_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                use_rmsnorm=use_rmsnorm,
            )
            for _ in range(middle_dense + trailing_dense)
        ])

        self.blocks = nn.ModuleList(blocks)
        self.norm = (RMSNorm(dec_dim) if use_rmsnorm else nn.LayerNorm(dec_dim))

    def forward(self, x_enc: torch.Tensor, x_detail: torch.Tensor) -> torch.Tensor:
        xg = self.enc_to_dec(x_enc)
        xd = x_detail

        for blk in self.blocks:
            if self.grad_checkpointing and self.training:
                xg, xd = checkpoint.checkpoint(blk, xg, xd, use_reentrant=False)
            else:
                xg, xd = blk(xg, xd)

        x = self.norm(xg + xd)
        return x  # (B, N, dec_dim)


@dataclass
class SkinSegFormerConfig:
    in_chans: int = 3
    num_classes: int = 1

    # Internal canvas size is fixed by design at 512.
    patch_size: int = 16  # 512/16 = 32 tokens/side

    # Encoder (ViT-like)
    enc_dim: int = 384
    enc_depth: int = 12
    enc_heads: int = 6
    enc_mlp_ratio: float = 4.0

    # Decoder (transformer-based)
    dec_dim: int = 256
    dec_depth: int = 6
    # The MoE depth is at most dec_depth - 2, as the MoE blocks are wrapped in dense layers.
    dec_moe_depth: int = 4
    dec_heads: int = 8
    dec_mlp_ratio: float = 4.0
    dec_num_experts: int = 8
    dec_moe_top_k: int = 2
    dec_moe_mlp_ratio: float = 1.75

    drop: float = 0.05
    attn_drop: float = 0.01

    use_abs_pos: bool = True
    use_rmsnorm: bool = True
    grad_checkpointing: bool = False


class SkinSegFormer(nn.Module):
    """
    Trainable 512-canvas ViT encoder + BiRefine transformer decoder + trainable upsample head.
    """
    def __init__(self, cfg: SkinSegFormerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        mean_t = torch.tensor([0.4956035, 0.45403906, 0.4229], dtype=torch.float32).view(1, -1, 1, 1)
        std_t = torch.tensor([0.35209113, 0.32731546, 0.32717964], dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("mean", mean_t)
        self.register_buffer("std", std_t)

        self.canvas = InterpCanvasProjector(cfg.in_chans, stem_ch=64)

        # ViT runs on the 512x512 canvas features (64 channels).
        self.encoder = ViTEncoder(
            in_chans=64,
            embed_dim=cfg.enc_dim,
            depth=cfg.enc_depth,
            num_heads=cfg.enc_heads,
            patch_size=cfg.patch_size,
            mlp_ratio=cfg.enc_mlp_ratio,
            drop=cfg.drop,
            attn_drop=cfg.attn_drop,
            use_abs_pos=cfg.use_abs_pos,
            use_rmsnorm=cfg.use_rmsnorm,
            grad_checkpointing=cfg.grad_checkpointing,
        )

        # Detail tokens come from high-res stem features (64 channels) to match token grid.
        self.detail = DetailTokenExtractor(in_chans=64, out_dim=cfg.dec_dim, patch_size=cfg.patch_size)

        self.decoder = TransformerDecoder(
            enc_dim=cfg.enc_dim,
            dec_dim=cfg.dec_dim,
            depth=cfg.dec_depth,
            num_heads=cfg.dec_heads,
            mlp_ratio=cfg.dec_mlp_ratio,
            drop=cfg.drop,
            attn_drop=cfg.attn_drop,
            moe_depth=cfg.dec_moe_depth,
            num_experts=cfg.dec_num_experts,
            moe_top_k=cfg.dec_moe_top_k,
            moe_mlp_ratio=cfg.dec_moe_mlp_ratio,
            use_rmsnorm=cfg.use_rmsnorm,
            grad_checkpointing=cfg.grad_checkpointing,
        )

        self.head = MidResRefineHead(
            dec_dim=cfg.dec_dim,
            num_classes=cfg.num_classes,
            patch_grid=32,
            mid_feat_ch=64,
            fuse_ch=128,
            refine_blocks=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input image tensor (B, C, H, W), H/W in {512, 1024}.
        :returns: Logits (B, num_classes, H, W).
        """
        bsz, ch, h, w = x.shape
        if not (h == w == 512 or h == w == 1024):
            raise ValueError(f"Only 512x512 or 1024x1024 supported, got {h}x{w}.")

        x_512, feat_mid = self.canvas((x - self.mean) / self.std)
        x_enc = self.encoder(x_512)
        x_detail = self.detail(feat_mid)
        x_dec = self.decoder(x_enc, x_detail)
        logits = self.head(x_dec, feat_mid, out_hw=(h, w))
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_force_explicit_attn(module: nn.Module, enabled: bool) -> None:
    for m in module.modules():
        if isinstance(m, MultiheadSelfAttention) or isinstance(m, MultiheadCrossAttention):
            m.force_explicit = enabled


def count_flops_forward(model: torch.nn.Module, *inputs, **kwargs) -> int:
    """
    Counts the FLOPs for the forward pass of a model.
    :param model: The model to evaluate.
    :param inputs: The inputs to the model.
    :param kwargs: Keyword arguments for the model.
    :returns: The number of FLOPs for the forward pass.
    """
    model.eval()
    # Use no_grad, not inference_mode (FlopCounterMode can report 0 under inference_mode).
    # See PyTorch issue discussion for details.
    with torch.no_grad():
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            _ = model(*inputs, **kwargs)
    return int(flop_counter.get_total_flops())


if __name__ == "__main__":
    cfg = SkinSegFormerConfig(
        num_classes=1,
        enc_dim=384,
        enc_depth=12,
        enc_heads=6,
        dec_dim=256,
        dec_depth=6,
        dec_heads=8,
    )
    model = SkinSegFormer(cfg)
    model.eval()

    x = torch.randn(2, 3, 1024, 1024)
    flop_test_tensor = torch.randn(1, 3, 1024, 1024)
    y = model(x)
    set_force_explicit_attn(model, True)
    flops = count_flops_forward(model, flop_test_tensor)
    print(
        f"logits: {y.shape}\n"
        f"params: {count_parameters(model):,}\n"
        f"FLOPs: {flops:,}"
    )
    exit(0)

    model = torch.compile(model, mode="max-autotune-no-cudagraphs").eval()
    _ = model(flop_test_tensor)

    start = time.perf_counter()
    for _ in range(10):
        model(flop_test_tensor)
    end = time.perf_counter()
    print(f"Time taken: {end - start:0.4f} seconds\n"
          f"{(end - start) / 10:0.4f} iterations per second")
