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
"""用於定義分數網路的通用層."""
import math
import string
from functools import partial
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from .normalization import ConditionalInstanceNorm2dPlus


def get_act(config):
    """根據 config 字串，回傳對應的激活函數（activation function）實例。"""

    # 如果指定 'elu'，回傳 PyTorch 的 ELU 激活函數
    if config == "elu":
        return nn.ELU()
    # 如果指定 'relu'，回傳 PyTorch 的 ReLU 激活函數
    elif config == "relu":
        return nn.ReLU()
    # 如果指定 'lrelu'，回傳 LeakyReLU，斜率設為 0.2（負半軸也允許小斜率通過）
    elif config == "lrelu":
        return nn.LeakyReLU(negative_slope=0.2)
    # 如果指定 'swish'，回傳 SiLU（Swish 函數在 PyTorch 中名稱為 SiLU）
    elif config == "swish":
        return nn.SiLU()
    # 若沒指定支援的名稱，則拋出例外
    else:
        raise NotImplementedError("activation function does not exist!")


def ncsn_conv1x1(
    in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0, padding=0
):
    """1x1 卷積層，與 NCSNv1/v2 架構一致。"""

    # 建立一個 2D 卷積層，kernel_size=1（等於通道混合、空間不變），
    # 可用於 channel 投影或特徵整合。
    conv = nn.Conv2d(
        in_planes,  # 輸入通道數
        out_planes,  # 輸出通道數
        kernel_size=1,  # 1x1 卷積
        stride=stride,  # 步幅（通常為1）
        bias=bias,  # 是否有偏置
        dilation=dilation,  # 膨脹係數
        padding=padding,  # padding 預設為0
    )

    # 若指定 init_scale=0，則設為極小值 1e-10，否則就用給定的 init_scale
    init_scale = 1e-10 if init_scale == 0 else init_scale
    # 權重做縮放初始化（原地乘以 init_scale）
    conv.weight.data *= init_scale
    # 偏置同樣縮放（如果有 bias）
    conv.bias.data *= init_scale
    # 回傳這個已初始化好的 1x1 卷積層
    return conv


def variance_scaling(
    scale, mode, distribution, in_axis=1, out_axis=0, dtype=torch.float32, device="cpu"
):
    """移植自 JAX 的 variance scaling 初始器，用於控制權重初始化的方差。

    Args:
        scale:         標準化比例，常見值有 1.0、2.0。
        mode:          方差計算方式：'fan_in', 'fan_out', 'fan_avg'。
        distribution:  分布型態：'normal'（高斯）或 'uniform'（均勻）。
        in_axis:       輸入通道維度（通常為 1）。
        out_axis:      輸出通道維度（通常為 0）。
        dtype:         權重型態，預設 torch.float32。
        device:        權重放置裝置，預設 'cpu'。
    Returns:
        返回一個根據 shape 給出初始化張量的函數。
    """

    # 計算 fan_in、fan_out
    def _compute_fans(shape, in_axis=1, out_axis=0):
        # receptive_field_size = 除去 in/out channel 的其餘維度乘積（kernel 寬高等）
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        # fan_in：每個輸入單位連到多少輸出
        fan_in = shape[in_axis] * receptive_field_size
        # fan_out：每個輸出單位連到多少輸入
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        # 用指定 shape 計算 fan_in 和 fan_out
        # fan_in: 每個神經元的輸入連結數（如 conv2d 的 in_channels * kernel_h * kernel_w）
        # fan_out: 每個神經元的輸出連結數（如 conv2d 的 out_channels * kernel_h * kernel_w）
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)

        # 根據初始化方法 mode，決定要用 fan_in 還是 fan_out 或兩者平均來當分母
        # 這會影響最終初始化張量的方差：
        # - "fan_in"（He初始化）：適合 ReLU 類激活函數，能防止前傳過程中方差不斷擴大
        # - "fan_out"（適合 softmax/分類器）：防止反向傳播梯度消失或爆炸
        # - "fan_avg"（Xavier初始化）：多層結構中平均處理訊號流動
        if mode == "fan_in":
            denominator = fan_in  # 用 fan_in 當分母（推薦 ReLU/LeakyReLU）
        elif mode == "fan_out":
            denominator = fan_out  # 用 fan_out 當分母（分類器/softmax 較常用）
        elif mode == "fan_avg":
            denominator = (
                fan_in + fan_out
            ) / 2  # 取平均，讓正反向訊號都穩定（適合 sigmoid/tanh）
        else:
            # 如果 mode 非以上三者，丟出錯誤提示
            raise ValueError(
                "invalid mode for variance scaling initializer: {}".format(mode)
            )

        # 計算方差
        variance = scale / denominator
        # 根據 distribution 回傳對應初始化權重
        if distribution == "normal":
            # 高斯分布，標準差 sqrt(variance)
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            # 均勻分布，範圍 [-sqrt(3*variance), sqrt(3*variance)]
            return (
                torch.rand(*shape, dtype=dtype, device=device) * 2.0 - 1.0
            ) * np.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    # 回傳一個可以根據 shape 產生初始化權重的函數
    return init


def default_init(scale=1.0):
    """採用與 DDPM 論文一致的預設權重初始化方式。"""
    # 如果 scale=0，則設為極小值 1e-10，避免初始化為全零（這會導致網路無法學習）
    scale = 1e-10 if scale == 0 else scale
    # 使用 variance_scaling，參數：
    # - scale: 方差比例（控制權重數值範圍大小）
    # - "fan_avg": 使用 fan_in 與 fan_out 的平均（即 Xavier/Glorot 初始化邏輯，適合平衡前向/反向訊號）
    # - "uniform": 均勻分布初始化，讓權重在 [-a, a] 之間隨機
    return variance_scaling(scale, "fan_avg", "uniform")


class Dense(nn.Module):
    """一個使用 default_init 權重初始化的全連接（線性）層。"""

    def __init__(self):
        super().__init__()


def ddpm_conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1.0, padding=0):
    """建立一個 1x1 卷積層，並使用 DDPM 推薦的初始化方式。"""

    # 建立 PyTorch 的 2D 卷積層，kernel_size=1（即 1x1 卷積），可用於特徵通道的線性轉換。
    conv = nn.Conv2d(
        in_planes,  # 輸入通道數
        out_planes,  # 輸出通道數
        kernel_size=1,  # 1x1 卷積
        stride=stride,  # 步幅
        padding=padding,  # padding 數量
        bias=bias,  # 是否加 bias
    )

    # 使用 DDPM 的 default_init 對權重做初始化，default_init 會根據給定的 init_scale 做變異數縮放
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)

    # 將 bias 全部初始化為 0，這樣網路一開始比較穩定
    nn.init.zeros_(conv.bias)

    # 回傳這個已經初始化好的 1x1 卷積層
    return conv


def ncsn_conv3x3(
    in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0, padding=1
):
    """建立一個 3x3 卷積層（NCSNv1/NCSNv2 標準初始化）。"""

    # 如果 init_scale 為 0，則設定為極小值 1e-10，防止全零初始化
    init_scale = 1e-10 if init_scale == 0 else init_scale

    # 建立一個 PyTorch 的 3x3 2D 卷積層
    conv = nn.Conv2d(
        in_planes,  # 輸入通道數
        out_planes,  # 輸出通道數
        stride=stride,  # 步幅
        bias=bias,  # 是否有偏置
        dilation=dilation,  # 空洞率（膨脹卷積）
        padding=padding,  # padding 數量（通常=1 保持空間尺寸不變）
        kernel_size=3,  # 3x3 卷積
    )

    # 權重做縮放初始化（原地乘上 init_scale）
    conv.weight.data *= init_scale
    # 偏置也做縮放
    conv.bias.data *= init_scale

    # 回傳這個初始化後的卷積層
    return conv


def ddpm_conv3x3(
    in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0, padding=1
):
    """建立一個 3x3 卷積層，並用 DDPM 論文推薦的初始化方式。"""

    # 建立 PyTorch 的 3x3 2D 卷積層
    conv = nn.Conv2d(
        in_planes,  # 輸入通道數
        out_planes,  # 輸出通道數
        kernel_size=3,  # 3x3 卷積
        stride=stride,  # 步幅
        padding=padding,  # padding 數量（預設=1，保持空間尺寸不變）
        dilation=dilation,  # 空洞率（可選，預設為1）
        bias=bias,  # 是否加偏置
    )

    # 使用 default_init（DDPM風格初始化）初始化權重
    # default_init(init_scale) 回傳一個初始化器，給定 shape 產生初始化權重
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)

    # 將 bias 初始化為全 0（讓網路起始時沒有偏移）
    nn.init.zeros_(conv.bias)

    # 回傳已經初始化好的卷積層
    return conv

    ###########################################################################
    # 以下函數從 NCSNv1/NCSNv2 程式碼庫移植過來:
    # https://github.com/ermongroup/ncsn
    # https://github.com/ermongroup/ncsnv2
    ###########################################################################


class CRPBlock(nn.Module):
    def __init__(self, features, n_stages, act=nn.ReLU(), maxpool=True):
        super().__init__()
        # 建立多個 3x3 卷積層，每一層不改變特徵圖通道數
        self.convs = nn.ModuleList()
        for i in range(n_stages):
            # 每一個 stage 都有一個 3x3 卷積
            self.convs.append(ncsn_conv3x3(features, features, stride=1, bias=False))
        self.n_stages = n_stages
        # 選擇用最大池化還是平均池化做 pooling（預設為 MaxPool）
        if maxpool:
            self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        else:
            self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        # 設定激活函數
        self.act = act

    def forward(self, x):
        # 先經過激活函數
        x = self.act(x)
        path = x
        # CRP 的核心：每個 stage 做一次 pooling -> conv，再與原始 x 相加（殘差疊加）
        for i in range(self.n_stages):
            path = self.pool(path)  # 先做池化（提取更大感受野的特徵）
            path = self.convs[i](path)  # 再經 3x3 卷積（特徵抽取）
            x = path + x  # 跟前一次的 x 做累加（累積 refinement）
        # 回傳累加 refinement 結果
        return x


class CondCRPBlock(nn.Module):
    def __init__(self, features, n_stages, num_classes, normalizer, act=nn.ReLU()):
        super().__init__()
        # 建立多個條件 normalization 層與 3x3 卷積層
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.normalizer = (
            normalizer  # 指定條件 normalization 類別（例如 ConditionalInstanceNorm2d）
        )
        for i in range(n_stages):
            # 每一層都先做條件 normalization（依照 class label 做歸一化）
            self.norms.append(normalizer(features, num_classes, bias=True))
            # 再接一個 3x3 卷積
            self.convs.append(ncsn_conv3x3(features, features, stride=1, bias=False))

        self.n_stages = n_stages
        # 這裡固定用平均池化（kernel 5x5, stride=1, padding=2），更平滑（不同於 CRPBlock 可以選 Max/Avg）
        self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        # 設定激活函數
        self.act = act

    def forward(self, x, y):
        # 先做一次激活
        x = self.act(x)
        path = x
        # 每一層：條件 normalization → 平均池化 → 卷積 → 累加
        for i in range(self.n_stages):
            path = self.norms[i](path, y)  # 根據 y（類別）做條件 normalization
            path = self.pool(path)  # 池化（擴大感受野）
            path = self.convs[i](path)  # 卷積
            x = path + x  # 殘差加總（多尺度融合）
        # 回傳最終融合結果
        return x


class RCUBlock(nn.Module):
    def __init__(self, features, n_blocks, n_stages, act=nn.ReLU()):
        super().__init__()

        # 動態建立多層 3x3 卷積（用 setattr 動態命名）
        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(
                    self,
                    "{}_{}_conv".format(i + 1, j + 1),  # 例如 1_1_conv、2_2_conv
                    ncsn_conv3x3(features, features, stride=1, bias=False),
                )

        self.stride = 1  # 預設 stride=1
        self.n_blocks = n_blocks  # 幾個 residual block（幾層殘差）
        self.n_stages = n_stages  # 每個 block 有幾層 convolution
        self.act = act  # 激活函數

    def forward(self, x):
        # 疊代每一個 residual block
        for i in range(self.n_blocks):
            residual = x  # 保留這一層的輸入 x
            # 每個 block 疊代多層 conv
            for j in range(self.n_stages):
                x = self.act(x)  # 激活函數
                # 動態取出對應的 conv 層（如 1_1_conv, 1_2_conv ...）
                x = getattr(self, "{}_{}_conv".format(i + 1, j + 1))(x)

            # block 結束後，把殘差加回來（residual connection）
            x += residual
        # 回傳多層殘差堆疊後的輸出
        return x


class CondRCUBlock(nn.Module):
    def __init__(
        self, features, n_blocks, n_stages, num_classes, normalizer, act=nn.ReLU()
    ):
        super().__init__()

        # 動態建立 n_blocks * n_stages 層的條件 normalizer 與卷積層
        for i in range(n_blocks):
            for j in range(n_stages):
                # 為每個 stage 加一個條件 normalization（根據 y 決定歸一化方式）
                setattr(
                    self,
                    "{}_{}_norm".format(i + 1, j + 1),
                    normalizer(features, num_classes, bias=True),
                )
                # 再加一個 3x3 卷積層
                setattr(
                    self,
                    "{}_{}_conv".format(i + 1, j + 1),
                    ncsn_conv3x3(features, features, stride=1, bias=False),
                )

        self.stride = 1
        self.n_blocks = n_blocks  # 殘差大區塊數
        self.n_stages = n_stages  # 每個區塊內有幾層
        self.act = act  # 激活函數
        self.normalizer = normalizer  # 條件正規化方法

    def forward(self, x, y):
        # 疊代每個 residual block
        for i in range(self.n_blocks):
            residual = x  # 保存這一區塊的輸入
            # 疊代每個 stage
            for j in range(self.n_stages):
                # 條件 normalization，根據 y（如類別/條件）調整歸一化
                x = getattr(self, "{}_{}_norm".format(i + 1, j + 1))(x, y)
                x = self.act(x)  # 激活函數
                # 卷積
                x = getattr(self, "{}_{}_conv".format(i + 1, j + 1))(x)
            # 殘差連接，把區塊起點加回來
            x += residual
        # 回傳最終結果
        return x


class MSFBlock(nn.Module):
    def __init__(self, in_planes, features):
        super().__init__()
        # in_planes 必須是 list 或 tuple，代表來自多個 stage 的特徵通道數
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)
        self.convs = nn.ModuleList()  # 儲存每個分支的卷積層
        self.features = features  # 最終融合特徵的通道數

        # 為每個輸入分支都建立一個 3x3 卷積，把各分支投影到同一個通道維度（features）
        for i in range(len(in_planes)):
            self.convs.append(ncsn_conv3x3(in_planes[i], features, stride=1, bias=True))

    def forward(self, xs, shape):
        # 建立一個全 0 的融合特徵張量（batch, features, H, W）
        sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
        # 疊代每一個分支（來自不同 stage/層的特徵）
        for i in range(len(self.convs)):
            h = self.convs[i](xs[i])  # 先做 3x3 卷積
            # 將這個分支的特徵 resize 到最終 shape（通常是最大解析度），
            # 用雙線性插值（bilinear）對齊空間大小
            h = F.interpolate(h, size=shape, mode="bilinear", align_corners=True)
            # 將每個分支加總，得到融合特徵圖
            sums += h
        # 回傳融合後的特徵圖
        return sums


class CondMSFBlock(nn.Module):
    def __init__(self, in_planes, features, num_classes, normalizer):
        super().__init__()
        # in_planes 必須是 list 或 tuple，代表多個來源特徵通道數
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)

        self.convs = nn.ModuleList()  # 儲存每個分支的卷積層
        self.norms = nn.ModuleList()  # 儲存每個分支的條件 normalization 層
        self.features = features  # 最終融合後的特徵通道數
        self.normalizer = normalizer  # 傳入的條件 normalization 類別

        # 為每個分支建立條件 normalization 和 3x3 卷積（將所有分支特徵映射到同一通道數）
        for i in range(len(in_planes)):
            self.convs.append(ncsn_conv3x3(in_planes[i], features, stride=1, bias=True))
            self.norms.append(normalizer(in_planes[i], num_classes, bias=True))

    def forward(self, xs, y, shape):
        # 建立一個全 0 融合特徵張量（batch, features, H, W）
        sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
        # 疊代每個分支（多層特徵）
        for i in range(len(self.convs)):
            h = self.norms[i](xs[i], y)  # 先對每個分支做條件 normalization
            h = self.convs[i](h)  # 然後 3x3 卷積調整通道
            h = F.interpolate(
                h, size=shape, mode="bilinear", align_corners=True
            )  # 空間對齊
            sums += h  # 融合加總
        # 回傳融合後的特徵圖
        return sums


class RefineBlock(nn.Module):
    def __init__(
        self, in_planes, features, act=nn.ReLU(), start=False, end=False, maxpool=True
    ):
        super().__init__()

        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)

        self.adapt_convs = nn.ModuleList()
        for i in range(n_blocks):
            self.adapt_convs.append(RCUBlock(in_planes[i], 2, 2, act))

        self.output_convs = RCUBlock(features, 3 if end else 1, 2, act)

        if not start:
            self.msf = MSFBlock(in_planes, features)

        self.crp = CRPBlock(features, 2, act, maxpool=maxpool)

    def forward(self, xs, output_shape):
        assert isinstance(xs, tuple) or isinstance(xs, list)
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i])
            hs.append(h)

        if self.n_blocks > 1:
            h = self.msf(hs, output_shape)
        else:
            h = hs[0]

        h = self.crp(h)
        h = self.output_convs(h)

        return h


class CondRefineBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        features,
        num_classes,
        normalizer,
        act=nn.ReLU(),
        start=False,
        end=False,
    ):
        super().__init__()

        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)

        self.adapt_convs = nn.ModuleList()
        for i in range(n_blocks):
            self.adapt_convs.append(
                CondRCUBlock(in_planes[i], 2, 2, num_classes, normalizer, act)
            )

        self.output_convs = CondRCUBlock(
            features, 3 if end else 1, 2, num_classes, normalizer, act
        )

        if not start:
            self.msf = CondMSFBlock(in_planes, features, num_classes, normalizer)

        self.crp = CondCRPBlock(features, 2, num_classes, normalizer, act)

    def forward(self, xs, y, output_shape):
        assert isinstance(xs, tuple) or isinstance(xs, list)
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i], y)
            hs.append(h)

        if self.n_blocks > 1:
            h = self.msf(hs, y, output_shape)
        else:
            h = hs[0]

        h = self.crp(h, y)
        h = self.output_convs(h, y)

        return h


class ConvMeanPool(nn.Module):
    def __init__(
        self, input_dim, output_dim, kernel_size=3, biases=True, adjust_padding=False
    ):
        super().__init__()
        if not adjust_padding:
            conv = nn.Conv2d(
                input_dim,
                output_dim,
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=biases,
            )
            self.conv = conv
        else:
            conv = nn.Conv2d(
                input_dim,
                output_dim,
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=biases,
            )

            self.conv = nn.Sequential(nn.ZeroPad2d((1, 0, 1, 0)), conv)

    def forward(self, inputs):
        output = self.conv(inputs)
        output = (
            sum(
                [
                    output[:, :, ::2, ::2],
                    output[:, :, 1::2, ::2],
                    output[:, :, ::2, 1::2],
                    output[:, :, 1::2, 1::2],
                ]
            )
            / 4.0
        )
        return output


class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim,
            output_dim,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=biases,
        )

    def forward(self, inputs):
        output = inputs
        output = (
            sum(
                [
                    output[:, :, ::2, ::2],
                    output[:, :, 1::2, ::2],
                    output[:, :, ::2, 1::2],
                    output[:, :, 1::2, 1::2],
                ]
            )
            / 4.0
        )
        return self.conv(output)


class UpsampleConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim,
            output_dim,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=biases,
        )
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, inputs):
        output = inputs
        output = torch.cat([output, output, output, output], dim=1)
        output = self.pixelshuffle(output)
        return self.conv(output)


class ConditionalResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_classes,
        resample=1,
        act=nn.ELU(),
        normalization=ConditionalInstanceNorm2dPlus,
        adjust_padding=False,
        dilation=None,
    ):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        self.normalization = normalization
        if resample == "down":
            if dilation > 1:
                self.conv1 = ncsn_conv3x3(input_dim, input_dim, dilation=dilation)
                self.normalize2 = normalization(input_dim, num_classes)
                self.conv2 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
            else:
                self.conv1 = ncsn_conv3x3(input_dim, input_dim)
                self.normalize2 = normalization(input_dim, num_classes)
                self.conv2 = ConvMeanPool(
                    input_dim, output_dim, 3, adjust_padding=adjust_padding
                )
                conv_shortcut = partial(
                    ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding
                )

        elif resample is None:
            if dilation > 1:
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
                self.conv1 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
                self.normalize2 = normalization(output_dim, num_classes)
                self.conv2 = ncsn_conv3x3(output_dim, output_dim, dilation=dilation)
            else:
                conv_shortcut = nn.Conv2d
                self.conv1 = ncsn_conv3x3(input_dim, output_dim)
                self.normalize2 = normalization(output_dim, num_classes)
                self.conv2 = ncsn_conv3x3(output_dim, output_dim)
        else:
            raise Exception("invalid resample value")

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)

        self.normalize1 = normalization(input_dim, num_classes)

    def forward(self, x, y):
        output = self.normalize1(x, y)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output, y)
        output = self.non_linearity(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)

        return shortcut + output


class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        resample=None,
        act=nn.ELU(),
        normalization=nn.InstanceNorm2d,
        adjust_padding=False,
        dilation=1,
    ):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        self.normalization = normalization
        if resample == "down":
            if dilation > 1:
                self.conv1 = ncsn_conv3x3(input_dim, input_dim, dilation=dilation)
                self.normalize2 = normalization(input_dim)
                self.conv2 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
            else:
                self.conv1 = ncsn_conv3x3(input_dim, input_dim)
                self.normalize2 = normalization(input_dim)
                self.conv2 = ConvMeanPool(
                    input_dim, output_dim, 3, adjust_padding=adjust_padding
                )
                conv_shortcut = partial(
                    ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding
                )

        elif resample is None:
            if dilation > 1:
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
                self.conv1 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
                self.normalize2 = normalization(output_dim)
                self.conv2 = ncsn_conv3x3(output_dim, output_dim, dilation=dilation)
            else:
                # conv_shortcut = nn.Conv2d ### Something wierd here.
                conv_shortcut = partial(ncsn_conv1x1)
                self.conv1 = ncsn_conv3x3(input_dim, output_dim)
                self.normalize2 = normalization(output_dim)
                self.conv2 = ncsn_conv3x3(output_dim, output_dim)
        else:
            raise Exception("invalid resample value")

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)

        self.normalize1 = normalization(input_dim)

    def forward(self, x):
        output = self.normalize1(x)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output)
        output = self.non_linearity(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)

        return shortcut + output


###########################################################################
# Functions below are ported over from the DDPM codebase:
#  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
###########################################################################


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def _einsum(a, b, c, x, y):
    einsum_str = "{},{}->{}".format("".join(a), "".join(b), "".join(c))
    return torch.einsum(einsum_str, x, y)


def contract_inner(x, y):
    """tensordot(x, y, 1)."""
    x_chars = list(string.ascii_lowercase[: len(x.shape)])
    y_chars = list(string.ascii_lowercase[len(x.shape) : len(y.shape) + len(x.shape)])
    y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
    out_chars = x_chars[:-1] + y_chars[1:]
    return _einsum(x_chars, y_chars, out_chars, x, y)


class NIN(nn.Module):
    def __init__(self, in_dim, num_units, init_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(
            default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True
        )
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        y = contract_inner(x, self.W) + self.b
        return y.permute(0, 3, 1, 2)


class AttnBlock(nn.Module):
    """Channel-wise self-attention block."""

    def __init__(self, channels):
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=0.0)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)

        w = torch.einsum("bchw,bcij->bhwij", q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum("bhwij,bcij->bchw", w, v)
        h = self.NIN_3(h)
        return x + h


class Upsample(nn.Module):
    def __init__(self, channels, with_conv=False):
        super().__init__()
        if with_conv:
            self.Conv_0 = ddpm_conv3x3(channels, channels)
        self.with_conv = with_conv

    def forward(self, x):
        B, C, H, W = x.shape
        h = F.interpolate(x, (H * 2, W * 2), mode="nearest")
        if self.with_conv:
            h = self.Conv_0(h)
        return h


class Downsample(nn.Module):
    def __init__(self, channels, with_conv=False):
        super().__init__()
        if with_conv:
            self.Conv_0 = ddpm_conv3x3(channels, channels, stride=2, padding=0)
        self.with_conv = with_conv

    def forward(self, x):
        B, C, H, W = x.shape
        # Emulate 'SAME' padding
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1))
            x = self.Conv_0(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)

        assert x.shape == (B, C, H // 2, W // 2)
        return x


class ResnetBlockDDPM(nn.Module):
    """The ResNet Blocks used in DDPM."""

    def __init__(
        self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False, dropout=0.1
    ):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6)
        self.act = act
        self.Conv_0 = ddpm_conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)

        self.GroupNorm_1 = nn.GroupNorm(num_groups=32, num_channels=out_ch, eps=1e-6)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = ddpm_conv3x3(out_ch, out_ch, init_scale=0.0)
        if in_ch != out_ch:
            if conv_shortcut:
                self.Conv_2 = ddpm_conv3x3(in_ch, out_ch)
            else:
                self.NIN_0 = NIN(in_ch, out_ch)
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.conv_shortcut = conv_shortcut

    def forward(self, x, temb=None):
        B, C, H, W = x.shape
        assert C == self.in_ch
        out_ch = self.out_ch if self.out_ch else self.in_ch
        h = self.act(self.GroupNorm_0(x))
        h = self.Conv_0(h)
        # Add bias to each feature map conditioned on the time embedding
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if C != out_ch:
            if self.conv_shortcut:
                x = self.Conv_2(x)
            else:
                x = self.NIN_0(x)
        return x + h
