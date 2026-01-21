import torch
import torch.nn as nn
import torch.nn.functional as F


class QuadrangleAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0,
                 window_size=7, rpe='v2', coords_lambda=20):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

        # Optional: You can add RPE or adaptive sampling modules here

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size

        # pad input if not divisible by window size
        pad_b = (ws - H % ws) % ws
        pad_r = (ws - W % ws) % ws
        x = F.pad(x, (0, pad_r, 0, pad_b))  # pad right and bottom
        _, _, Hp, Wp = x.shape

        # QKV projection and reshape
        qkv = self.qkv(x)  # (B, 3C, Hp, Wp)
        q, k, v = qkv.chunk(3, dim=1)  # each is (B, C, Hp, Wp)

        # reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim, -1).transpose(2, 3)  # (B, h, N, head_dim)
        k = k.view(B, self.num_heads, self.head_dim, -1)  # (B, h, head_dim, N)
        v = v.view(B, self.num_heads, self.head_dim, -1).transpose(2, 3)  # (B, h, N, head_dim)

        attn = (q @ k) * self.scale  # (B, h, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B, h, N, head_dim)
        out = out.transpose(2, 3).reshape(B, C, Hp, Wp)  # (B, C, Hp, Wp)

        out = self.proj(out)
        out = self.proj_drop(out)

        return out[:, :, :H, :W]  # crop back to original size
