# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Layers for defining NCSN++."""
from . import layers
from . import up_or_down_sampling
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

conv1x1 = layers.ddpm_conv1x1
conv3x3 = layers.ddpm_conv3x3
NIN = layers.NIN
default_init = layers.default_init


class GaussianFourierProjection(nn.Module):
    """給 time/noise embedding 用的高斯傅立葉嵌入(Gaussian Fourier embedding)，常見於擴散模型，把 scalar timestep 嵌入為高維向量"""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        # 建立一個長度為 embedding_size 的高斯分佈隨機向量
        # 這個向量會用來做傅立葉嵌入(Fourier embedding)的權重（每個維度都隨機），
        # 通常用來把 scalar 的時間步 t 映射到高維空間（像是位置嵌入 positional embedding）
        # 設定 requires_grad=False，表示這個參數在訓練時不會被優化（保持固定），
        # 因為傅立葉嵌入(Fourier embedding)的理論就是希望用隨機常數、而不是學習式的 embedding
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        # x 通常是 [batch,] 或 [batch, 1]，代表每個 batch 的 scalar 時間步（例如 diffusion 的 t）

        # 將 x 維度擴展，乘上 self.W 得到一個 [batch, embedding_size] 的矩陣，
        # 每一列都代表將 x 的 scalar 值用不同隨機權重做投影
        # 乘上 2*pi 是為了讓 sin/cos 的 argument 覆蓋一個完整的週期（傅立葉理論）
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi

        # 將上一步的投影經過 sin 和 cos，得到兩個 [batch, embedding_size] 的向量
        # sin/cos 可以讓模型辨識不同的時間位置、並讓 embedding 有豐富的變化性
        # 最後在最後一個維度（特徵維）將 sin 和 cos 結果串接起來，變成 [batch, 2*embedding_size]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Combine(nn.Module):
    """用來融合 Encoder-Decoder 跳接 (skip connection) 特徵的方法，有 'cat'（串接）和 'sum'（相加）兩種。"""

    def __init__(self, dim1, dim2, method="cat"):
        super().__init__()
        # 這裡建立一個 1x1 卷積，將輸入 x 的通道數 dim1 轉換成 dim2
        # 主要目的是讓 x 的通道數和 y 一致，好方便後續進行融合
        self.Conv_0 = conv1x1(dim1, dim2)
        # 設定融合方法，"cat" 代表串接，"sum" 代表逐元素相加
        self.method = method

    def forward(self, x, y):
        # 先將 x 經過 1x1 卷積，把通道數轉成 dim2
        h = self.Conv_0(x)
        # 如果指定 method 為 "cat"（串接），則沿著通道維度 (dim=1) 串接 h 和 y
        if self.method == "cat":
            return torch.cat([h, y], dim=1)
        # 如果指定 method 為 "sum"（相加），則直接將 h 和 y 做逐元素加總
        elif self.method == "sum":
            return h + y
        # 如果指定的融合方式不是 "cat" 或 "sum"，就拋出錯誤
        else:
            raise ValueError(f"Method {self.method} not recognized.")


class AttnBlockpp(nn.Module):
    """通道自注意力（self-attention）模組。可捕捉全域特徵關聯性，用於 U-Net 類擴散模型的關鍵積木。"""

    def __init__(self, channels, skip_rescale=False, init_scale=0.0):
        super().__init__()
        # 使用 GroupNorm 做正規化，有助於模型穩定與收斂
        # num_groups 最多 32，否則以 channels // 4 為準，eps 避免數值不穩
        self.GroupNorm_0 = nn.GroupNorm(
            num_groups=min(channels // 4, 32), num_channels=channels, eps=1e-6
        )
        # 以下四個 NIN（Network-in-Network）全連接層，對每個像素點進行通道變換（等價於 1x1 卷積）
        self.NIN_0 = NIN(channels, channels)  # 給 q (query) 用
        self.NIN_1 = NIN(channels, channels)  # 給 k (key) 用
        self.NIN_2 = NIN(channels, channels)  # 給 v (value) 用
        self.NIN_3 = NIN(channels, channels, init_scale=init_scale)  # 用於最後輸出殘差
        self.skip_rescale = (
            skip_rescale  # 是否啟用 skip connection rescale（根據論文選擇）
        )

    def forward(self, x):
        B, C, H, W = x.shape  # B: batch size, C: channels, H/W: 高/寬
        # 先做 group normalization，讓不同 batch 資料分布穩定
        h = self.GroupNorm_0(x)
        # 分別對正規化後的特徵做線性投影，取得 query、key、value
        q = self.NIN_0(h)  # query，形狀仍為 [B, C, H, W]
        k = self.NIN_1(h)  # key
        v = self.NIN_2(h)  # value
        # 計算注意力權重 w（q 與 k 進行廣播內積），再做縮放
        w = torch.einsum("bchw,bcij->bhwij", q, k) * (int(C) ** (-0.5))
        # 把權重展平成 (B, H, W, H*W)，每個 pixel 可以看到全圖
        w = torch.reshape(w, (B, H, W, H * W))
        # 使用 softmax 正規化最後一維，使注意力權重為機率分布
        w = F.softmax(w, dim=-1)
        # 還原回 (B, H, W, H, W) 的五維權重
        w = torch.reshape(w, (B, H, W, H, W))
        # 將權重 w 應用到 v（value），等於加權求和，類似 transformer 的 self-attention
        h = torch.einsum("bhwij,bcij->bchw", w, v)
        # 最後再經過一層 NIN 做殘差轉換
        h = self.NIN_3(h)
        # 決定回傳型式：是否對 skip connection 做 rescale，預設就是直接加
        if not self.skip_rescale:
            return x + h  # 殘差連接，直接相加
        else:
            return (x + h) / np.sqrt(2.0)  # 也可以用論文推薦的 rescale，讓殘差穩定


class Upsample(nn.Module):
    """
    上採樣模組。支援最近鄰插值或 FIR kernel（高品質上採樣），
    可以選擇要不要在上採樣後加卷積層（with_conv）。
    """

    def __init__(
        self,
        in_ch=None,  # 輸入通道數
        out_ch=None,  # 輸出通道數（可選，預設跟 in_ch 一樣）
        with_conv=False,  # 是否在上採樣後加卷積層
        fir=False,  # 是否使用 FIR kernel 上採樣（更高品質）
        fir_kernel=(1, 3, 3, 1),  # FIR 上採樣的 kernel
    ):
        super().__init__()
        # 如果沒有設定 out_ch，預設跟 in_ch 一樣
        out_ch = out_ch if out_ch else in_ch
        # 如果沒選 fir，採用最近鄰插值方式上採樣
        if not fir:
            if with_conv:
                # 如果需要卷積，則用一個 3x3 卷積（通常用於特徵調整）
                self.Conv_0 = conv3x3(in_ch, out_ch)
        else:
            # 如果使用 FIR kernel 上採樣
            if with_conv:
                # 建立一個自定義的卷積上採樣層（內含 upsampling 和卷積）
                self.Conv2d_0 = up_or_down_sampling.Conv2d(
                    in_ch,
                    out_ch,
                    kernel=3,
                    up=True,  # up=True 代表上採樣
                    resample_kernel=fir_kernel,  # 指定 FIR kernel
                    use_bias=True,
                    kernel_init=default_init(),
                )
        # 紀錄參數
        self.fir = fir
        self.with_conv = with_conv
        self.fir_kernel = fir_kernel
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape  # B: batch, C: channels, H: height, W: width
        if not self.fir:
            # 不用 FIR kernel，就用最近鄰插值方式將 H/W 變 2 倍
            h = F.interpolate(x, (H * 2, W * 2), "nearest")
            if self.with_conv:
                # 上採樣後再用 3x3 卷積處理特徵
                h = self.Conv_0(h)
        else:
            # 如果用有限脈衝響應濾波器(FIR kernel)
            if not self.with_conv:
                # 只上採樣不用卷積，直接呼叫自定義 FIR kernel 上採樣
                # FIR kernel 上採樣：先把影像空間拉大（多塞進 0 或插入空位），然後用一個濾波器（kernel）掃過整張圖做加權平均，這個 kernel 就是 FIR kernel。
                h = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
            else:
                # 用包含卷積的自定義 Conv2d_0 來做上採樣
                h = self.Conv2d_0(x)
        return h  # 回傳上採樣後的特徵圖


class Downsample(nn.Module):
    """
    下採樣（降解析度）模組。支援一般平均池化、或高品質 FIR kernel 濾波下採樣，
    也可以選擇要不要用卷積（with_conv）來學習下採樣特徵。
    """

    def __init__(
        self,
        in_ch=None,  # 輸入通道數
        out_ch=None,  # 輸出通道數（預設跟 in_ch 相同）
        with_conv=False,  # 是否使用卷積下採樣
        fir=False,  # 是否用 FIR kernel 下採樣
        fir_kernel=(1, 3, 3, 1),  # FIR kernel 參數
    ):
        super().__init__()
        # 如果沒有指定 out_ch，預設跟 in_ch 一樣
        out_ch = out_ch if out_ch else in_ch
        # 如果不用 FIR kernel
        if not fir:
            if with_conv:
                # 用 3x3 卷積、stride=2 直接做下採樣
                # padding=0 是因為後面手動 pad
                self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)
        else:
            # 用自定義的 Conv2d_0 層（可同時做 FIR 濾波與下採樣）
            if with_conv:
                self.Conv2d_0 = up_or_down_sampling.Conv2d(
                    in_ch,
                    out_ch,
                    kernel=3,
                    down=True,  # down=True 代表做下採樣
                    resample_kernel=fir_kernel,  # 指定 FIR kernel
                    use_bias=True,
                    kernel_init=default_init(),
                )
        # 記錄初始化參數
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.with_conv = with_conv
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape  # B: batch, C: channels, H/W: 空間維度
        if not self.fir:
            if self.with_conv:
                # 若使用卷積，先對右/下各 pad 1（讓卷積輸出 shape 對齊），再用 stride=2 卷積下採樣
                x = F.pad(x, (0, 1, 0, 1))
                x = self.Conv_0(x)
            else:
                # 不用卷積，直接用 2x2 平均池化下採樣
                x = F.avg_pool2d(x, 2, stride=2)
        else:
            if not self.with_conv:
                # 只用 FIR kernel 濾波+下採樣（不學習參數）
                x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
            else:
                # 用自定義的卷積+FIR kernel 濾波一起下採樣（有學習參數）
                x = self.Conv2d_0(x)

        return x  # 回傳下採樣後的特徵圖


class ResnetBlockDDPMpp(nn.Module):
    """ResBlock adapted from DDPM。可用於 U-Net/擴散模型的殘差區塊。"""

    def __init__(
        self,
        act,  # 激活函數 (如 Swish、ReLU、ELU)
        in_ch,  # 輸入通道數
        out_ch=None,  # 輸出通道數，預設和 in_ch 相同
        temb_dim=None,  # 時間 embedding 向量維度（可選，給擴散 t embedding 用）
        conv_shortcut=False,  # 如果 in/out channel 不同，是否用卷積做捷徑
        dropout=0.1,  # dropout 比例
        skip_rescale=False,  # 殘差是否 rescale
        init_scale=0.0,  # 初始化縮放比例
    ):
        super().__init__()
        # 預設 out_ch = in_ch
        out_ch = out_ch if out_ch else in_ch

        # 第一個 group normalization，穩定 batch 分布
        # GN 會把一個 batch 裡的每個樣本的特徵通道切成 num_groups 組，分組做均值與標準差歸一化，
        # 能兼具 BatchNorm 和 LayerNorm 優點
        self.GroupNorm_0 = nn.GroupNorm(
            num_groups=min(
                in_ch // 4, 32
            ),  # 決定分幾組，每組的 channel 數盡量約為 4，但不超過 32 組
            num_channels=in_ch,  # 輸入的總通道數（feature map channel 數）
            eps=1e-6,  # 很小的數字，防止分母為 0，提升數值穩定性
        )

        # 第一個 3x3 卷積（用於輸入轉換到 out_ch 維度）
        self.Conv_0 = conv3x3(in_ch, out_ch)

        # 若指定有 time embedding，就加一個線性層接收 temb，會加到主分支裡
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            # 權重用自定義初始化
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            # bias 初始化為 0
            nn.init.zeros_(self.Dense_0.bias)

        # 第二個 group normalization
        self.GroupNorm_1 = nn.GroupNorm(
            num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6
        )
        # dropout 層，用來隨機丟掉部分神經元，防止過擬合
        self.Dropout_0 = nn.Dropout(dropout)
        # 第二個 3x3 卷積，最後一層常加 init_scale 來穩定訓練
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)

        # 如果 in_ch ≠ out_ch，捷徑分支（shortcut）要做對齊
        if in_ch != out_ch:
            if conv_shortcut:
                # 用卷積來對齊 channel
                self.Conv_2 = conv3x3(in_ch, out_ch)
            else:
                # 或用 NIN (1x1 conv/全連接) 對齊 channel
                self.NIN_0 = NIN(in_ch, out_ch)

        self.skip_rescale = skip_rescale  # 是否做殘差 rescale
        self.act = act  # 激活函數
        self.out_ch = out_ch  # 輸出通道數
        self.conv_shortcut = conv_shortcut  # 捷徑分支是否用卷積

    def forward(self, x, temb=None):
        # 先對輸入做 group normalization，再經過激活函數
        h = self.act(self.GroupNorm_0(x))
        # 接著用 3x3 卷積做特徵轉換（可能調整 channel 數）
        h = self.Conv_0(h)
        # 如果有提供時間步 embedding（temb），則經過 Dense_0 處理後加到特徵圖上
        # self.act(temb)：先做激活函數；Dense_0：投影成 out_ch 維；[:, :, None, None]：擴展維度以便與 h 相加
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        # 再做一次 group normalization 和激活函數
        h = self.act(self.GroupNorm_1(h))
        # dropout，隨機丟棄部分神經元（防止過擬合）
        h = self.Dropout_0(h)
        # 再經過一層 3x3 卷積
        h = self.Conv_1(h)
        # 如果輸入 x 的通道數和目標輸出通道數不同，則對捷徑分支做 channel 對齊
        if x.shape[1] != self.out_ch:
            if self.conv_shortcut:
                # 用 3x3 卷積做捷徑分支對齊
                x = self.Conv_2(x)
            else:
                # 用 NIN (1x1 conv/全連接) 做捷徑分支對齊
                x = self.NIN_0(x)
        # 殘差連接，可選是否 rescale（根據 skip_rescale 設定）
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)


class ResnetBlockBigGANpp(nn.Module):
    """
    改良自 BigGAN 的 ResNet Block，可用於 U-Net/生成式模型，
    支援 upsampling, downsampling, time embedding、FIR kernel，以及通道對齊等。
    """

    def __init__(
        self,
        act,  # 激活函數（如 Swish、ReLU）
        in_ch,  # 輸入通道數
        out_ch=None,  # 輸出通道數（預設與 in_ch 相同）
        temb_dim=None,  # 時間 embedding 維度（可選）
        up=False,  # 是否做上採樣
        down=False,  # 是否做下採樣
        dropout=0.1,  # dropout 比例
        fir=False,  # 是否用 FIR kernel 做升/降採樣
        fir_kernel=(1, 3, 3, 1),  # FIR kernel
        skip_rescale=True,  # 殘差是否做 rescale
        init_scale=0.0,  # 初始化縮放因子
    ):
        super().__init__()

        out_ch = out_ch if out_ch else in_ch  # 沒指定就用 in_ch
        # 第一個 group normalization
        self.GroupNorm_0 = nn.GroupNorm(
            num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
        )
        # 記錄 up/down/fir/fir_kernel 參數
        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel

        # 第一個 3x3 卷積，負責主分支特徵轉換
        self.Conv_0 = conv3x3(in_ch, out_ch)
        # 如果指定有時間步 embedding，建立一個線性層 Dense_0
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            # 權重用指定方式初始化
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            # bias 設為 0
            nn.init.zeros_(self.Dense_0.bias)

        # 第二個 group normalization
        self.GroupNorm_1 = nn.GroupNorm(
            num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6
        )
        # dropout 隨機丟棄部分神經元
        self.Dropout_0 = nn.Dropout(dropout)
        # 第二個 3x3 卷積（主分支）
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        # 如果 in/out 通道不同，或有 up/down（升/降採樣），則捷徑分支需加 1x1 卷積對齊
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch)

        self.skip_rescale = skip_rescale  # 是否啟用殘差 rescale
        self.act = act  # 激活函數
        self.in_ch = in_ch  # 輸入通道數
        self.out_ch = out_ch  # 輸出通道數

    def forward(self, x, temb=None):
        # 先對輸入 x 做 group normalization，再經過激活函數
        h = self.act(self.GroupNorm_0(x))

        # 根據設定決定是否做上採樣
        if self.up:
            if self.fir:
                # 用高品質 FIR kernel 做上採樣（平滑+插值），h 和捷徑 x 都要上採樣
                h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
            else:
                # 用最簡單最近鄰上採樣，h 和捷徑 x 都上採樣
                h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
        # 如果不是 up 是 down，就做下採樣
        elif self.down:
            if self.fir:
                # 用高品質 FIR kernel 做下採樣（濾波+取樣）
                h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
            else:
                # 用最簡單平均池化做下採樣
                h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_downsample_2d(x, factor=2)

        # 主分支經過 3x3 卷積
        h = self.Conv_0(h)

        # 若有 time embedding，投影並加到主分支特徵圖上（會自動 broadcast 到空間維度）
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]

        # 主分支再經 group normalization、激活函數
        h = self.act(self.GroupNorm_1(h))
        # dropout 防止過擬合
        h = self.Dropout_0(h)
        # 最後再經過一層 3x3 卷積
        h = self.Conv_1(h)

        # 捷徑分支：如果 in/out channel 不同或有 up/down，則經 1x1 卷積做 channel 對齊
        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)

        # 殘差連接，可以選擇是否做 rescale（防止深網路梯度爆炸/消失）
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)
