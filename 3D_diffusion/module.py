import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv3d(in_ch, in_ch, 3, stride=2, padding=1)  # Project : conv2d -> conv3d
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv3d(in_ch, in_ch, 3, stride=1, padding=1)  # Project : conv2d -> conv3d
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        # _, _, D, H, W = x.shape ## Project ##
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(16, in_ch) ## Project ## 32->16
        self.proj_q = nn.Conv3d(in_ch, in_ch, 1, stride=1, padding=0)  # Project : conv2d -> conv3d
        self.proj_k = nn.Conv3d(in_ch, in_ch, 1, stride=1, padding=0)  # Project : conv2d -> conv3d
        self.proj_v = nn.Conv3d(in_ch, in_ch, 1, stride=1, padding=0)  # Project : conv2d -> conv3d
        self.proj = nn.Conv3d(in_ch, in_ch, 1, stride=1, padding=0)  # Project : conv2d -> conv3d
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        ####### Project #######
        B, C, D, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 4, 1).view(B, D * H * W, C)
        k = k.view(B, C, D * H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, D * H * W, D * H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 4, 1).view(B, D * H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, D * H * W, C]
        h = h.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        h = self.proj(h)
        ####### Project #######
        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_ch), ## Project ## 32->16
            Swish(),
            nn.Conv3d(in_ch, out_ch, 3, stride=1, padding=1),  # Project : conv2d -> conv3d
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_ch), ## Project ## 32->16
            Swish(),
            nn.Dropout(dropout),
            nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1),  # Project : conv2d -> conv3d
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv3d(in_ch, out_ch, 1, stride=1, padding=0)  # Project : conv2d -> conv3d
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        # print(h.shape)
        # print(self.temb_proj(temb).shape)
        # print(self.temb_proj(temb)[:, :, None, None, None].shape)
        h += self.temb_proj(temb)[:, :, None, None, None]  # Project : [:, :, None, None] -> [:, :, None, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        if t.ndim == 0:
            t = t.unsqueeze(-1)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
