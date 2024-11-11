from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from module import DownSample, ResBlock, Swish, TimeEmbedding, UpSample
from torch.nn import init
import time
# from torchsummary import summary
class UNet3D(nn.Module):
    def __init__(self, T=1000, voxel_resolution=128, ch=64, ch_mult=[1, 2, 2, 2], attn=[1], num_res_blocks=4,
                 dropout=0.1, use_cfg=False, cfg_dropout=0.1, num_classes=None):
        super().__init__()
        self.voxel_resolution = voxel_resolution  # Project
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        # self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.time_embedding = TimeEmbedding(tdim)

        # classifier-free guidance
        self.use_cfg = use_cfg
        self.cfg_dropout = cfg_dropout
        if use_cfg:
            assert num_classes is not None
            cdim = tdim
            self.class_embedding = nn.Embedding(num_classes + 1, cdim)

        self.down1 = DownSample(1)  # Project : downsampling at start (128->64)
        self.down2 = DownSample(1)  # Project : downsampling at start (64->32)
        self.head = nn.Conv3d(1, ch, kernel_size=3, stride=1, padding=1)  # Project : conv2d -> conv3d, input_channel 3->1
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(16, now_ch),  # Project : 32->16
            Swish(),
            nn.Conv3d(now_ch, 1, 3, stride=1, padding=1)  # Project : conv2d -> conv3d, output_channel 3->1
        )
        self.up1 = UpSample(1)  # Project : upsampling at the end (32->64)
        self.up2 = UpSample(1)  # Project : upsampling at the end (64->128)

        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, timestep, class_label=None):
        # Timestep embedding
        temb = self.time_embedding(timestep)

        if self.use_cfg and class_label is not None:
            if self.training:
                assert not torch.any(class_label == 0)  # 0 for null.

                ######## TODO ########
                # DO NOT change the code outside this part.
                # Assignment 2-2. Implement random null conditioning in CFG training.
                # class_label = class_label.to(self.device)
                null_rep = torch.zeros_like(class_label)
                class_label = torch.where(torch.rand_like(class_label) < self.cfg_dropout, null_rep, class_label)
                # class_label = class_label.to(self.device)

                #######################

            ######## TODO ########
            # DO NOT change the code outside this part.
            # Assignment 2-1. Implement class conditioning
            class_emb = self.class_embedding(class_label.to(timestep.device))
            temb = temb + class_emb
            #######################

        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)
        h = self.up1(h, temb)  # Project upsampling 32-> 64
        h = self.up2(h, temb)  # Project upsampling 64-> 128

        assert len(hs) == 0
        return h


if __name__=='__main__':
    print("GPU is", "available" if torch.cuda.is_available() else "not available")
    net = UNet3D(ch=16)
    net = net.to(device='cuda:0')
    x = torch.ones(16, 1, 32, 32, 32).to(device='cuda:0')
    ts = np.random.choice(np.arange(1000), 16)
    ts = torch.from_numpy(ts).to(device='cuda:0')
    start_time = time.time()
    print(net.forward(x, ts).size())
    # summary(model=model, input_size=(1, 128, 128, 128), batch_size=16, device="cuda:0")
    print("--- %s seconds ---" % (time.time() - start_time))
