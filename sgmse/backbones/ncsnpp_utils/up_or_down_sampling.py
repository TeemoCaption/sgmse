"""Layers used for up-sampling or down-sampling images.

Many functions are ported from https://github.com/NVlabs/stylegan2.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from .op import upfirdn2d


# Function ported from StyleGAN2
def get_weight(module, shape, weight_var="weight", kernel_init=None):
    """Get/create weight tensor for a convolution or fully-connected layer."""

    return module.param(weight_var, kernel_init, shape)


class Conv2d(nn.Module):
    """Conv2d layer with optimal upsampling and downsampling (StyleGAN2)."""

    def __init__(
        self,
        in_ch,
        out_ch,
        kernel,
        up=False,
        down=False,
        resample_kernel=(1, 3, 3, 1),
        use_bias=True,
        kernel_init=None,
    ):
        super().__init__()
        assert not (up and down)
        assert kernel >= 1 and kernel % 2 == 1
        self.weight = nn.Parameter(torch.zeros(out_ch, in_ch, kernel, kernel))
        if kernel_init is not None:
            self.weight.data = kernel_init(self.weight.data.shape)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_ch))

        self.up = up
        self.down = down
        self.resample_kernel = resample_kernel
        self.kernel = kernel
        self.use_bias = use_bias

    def forward(self, x):
        if self.up:
            x = upsample_conv_2d(x, self.weight, k=self.resample_kernel)
        elif self.down:
            x = conv_downsample_2d(x, self.weight, k=self.resample_kernel)
        else:
            x = F.conv2d(x, self.weight, stride=1, padding=self.kernel // 2)

        if self.use_bias:
            x = x + self.bias.reshape(1, -1, 1, 1)

        return x


def naive_upsample_2d(x, factor=2):
    # 假設 x 形狀是 [N, C, H, W]
    _N, C, H, W = x.shape

    # 先將 x reshape 為 [N, C, H, 1, W, 1]，
    # 在高（H）和寬（W）這兩個維度後面各加一個新的維度，方便後續在這些新維度上做複製
    x = torch.reshape(x, (-1, C, H, 1, W, 1))

    # 使用 repeat 把每一個像素在 H 軸後面的那個新維度複製 factor 次，
    # 在 W 軸後面的那個新維度也複製 factor 次
    # 這樣就相當於每個像素都擴展為 factor*factor 的小區塊（做最近鄰放大）
    x = x.repeat(1, 1, 1, factor, 1, factor)

    # 再將維度 reshape 回 [N, C, H*factor, W*factor]
    # 這樣高、寬都擴大 factor 倍，達到上採樣效果
    return torch.reshape(x, (-1, C, H * factor, W * factor))


def naive_downsample_2d(x, factor=2):
    # 假設 x 的 shape 是 [N, C, H, W]，分別是 batch, channel, 高, 寬
    _N, C, H, W = x.shape

    # 將 x reshape 成 [N, C, H // factor, factor, W // factor, factor]
    # 意思是把每 factor x factor 區域切成一小塊，方便後面做平均
    x = torch.reshape(x, (-1, C, H // factor, factor, W // factor, factor))

    # 在 factor 維度 (dim=3, dim=5) 上做平均，相當於將每 factor x factor 小區域平均為一個像素
    # 這就是最簡單的平均池化下採樣
    return torch.mean(x, dim=(3, 5))


def upsample_conv_2d(x, w, k=None, factor=2, gain=1):
    """Fused `upsample_2d()` followed by `tf.nn.conv2d()`.

    Padding is performed only once at the beginning, not between the
    operations.
    The fused op is considerably more efficient than performing the same
    calculation
    using standard TensorFlow ops. It supports gradients of arbitrary order.
    Args:
      x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
        C]`.
      w:            Weight tensor of the shape `[filterH, filterW, inChannels,
        outChannels]`. Grouped convolution can be performed by `inChannels =
        x.shape[0] // numGroups`.
      k:            FIR filter of the shape `[firH, firW]` or `[firN]`
        (separable). The default is `[1] * factor`, which corresponds to
        nearest-neighbor upsampling.
      factor:       Integer upsampling factor (default: 2).
      gain:         Scaling factor for signal magnitude (default: 1.0).

    Returns:
      Tensor of the shape `[N, C, H * factor, W * factor]` or
      `[N, H * factor, W * factor, C]`, and same datatype as `x`.
    """

    assert isinstance(factor, int) and factor >= 1

    # Check weight shape.
    assert len(w.shape) == 4
    convH = w.shape[2]
    convW = w.shape[3]
    inC = w.shape[1]
    outC = w.shape[0]

    assert convW == convH

    # Setup filter kernel.
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor**2))
    p = (k.shape[0] - factor) - (convW - 1)

    stride = (factor, factor)

    # Determine data dimensions.
    stride = [1, 1, factor, factor]
    output_shape = (
        (_shape(x, 2) - 1) * factor + convH,
        (_shape(x, 3) - 1) * factor + convW,
    )
    output_padding = (
        output_shape[0] - (_shape(x, 2) - 1) * stride[0] - convH,
        output_shape[1] - (_shape(x, 3) - 1) * stride[1] - convW,
    )
    assert output_padding[0] >= 0 and output_padding[1] >= 0
    num_groups = _shape(x, 1) // inC

    # Transpose weights.
    w = torch.reshape(w, (num_groups, -1, inC, convH, convW))
    w = w[..., ::-1, ::-1].permute(0, 2, 1, 3, 4)
    w = torch.reshape(w, (num_groups * inC, -1, convH, convW))

    x = F.conv_transpose2d(
        x, w, stride=stride, output_padding=output_padding, padding=0
    )
    ## Original TF code.
    # x = tf.nn.conv2d_transpose(
    #     x,
    #     w,
    #     output_shape=output_shape,
    #     strides=stride,
    #     padding='VALID',
    #     data_format=data_format)
    ## JAX equivalent

    return upfirdn2d(
        x, torch.tensor(k, device=x.device), pad=((p + 1) // 2 + factor - 1, p // 2 + 1)
    )


def conv_downsample_2d(x, w, k=None, factor=2, gain=1):
    """Fused `tf.nn.conv2d()` followed by `downsample_2d()`.

    Padding is performed only once at the beginning, not between the operations.
    The fused op is considerably more efficient than performing the same
    calculation
    using standard TensorFlow ops. It supports gradients of arbitrary order.
    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
          C]`.
        w:            Weight tensor of the shape `[filterH, filterW, inChannels,
          outChannels]`. Grouped convolution can be performed by `inChannels =
          x.shape[0] // numGroups`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to
          average pooling.
        factor:       Integer downsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H // factor, W // factor]` or
        `[N, H // factor, W // factor, C]`, and same datatype as `x`.
    """

    assert isinstance(factor, int) and factor >= 1
    _outC, _inC, convH, convW = w.shape
    assert convW == convH
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = (k.shape[0] - factor) + (convW - 1)
    s = [factor, factor]
    x = upfirdn2d(x, torch.tensor(k, device=x.device), pad=((p + 1) // 2, p // 2))
    return F.conv2d(x, w, stride=s, padding=0)


def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)
    assert k.ndim == 2
    assert k.shape[0] == k.shape[1]
    return k


def _shape(x, dim):
    return x.shape[dim]


def upsample_2d(x, k=None, factor=2, gain=1):
    r"""用指定的濾波器對一批二維影像進行上採樣.

    Args:
        x:    輸入的 4D tensor，shape 可以是 [N, C, H, W] 或 [N, H, W, C]，
              代表 batch, channel, 高度, 寬度（PyTorch 默認 [N, C, H, W]）。
        k:    FIR 濾波器（filter kernel），shape 可以是一維（[firN]）或二維（[firH, firW]）。
              預設是 [1]*factor，等同於最近鄰上採樣。
        factor: 上採樣倍率（預設 2，表示高/寬都放大兩倍）。
        gain:  影像強度縮放倍率，預設 1.0。
    Returns:
        回傳上採樣後的 tensor，shape 是 [N, C, H*factor, W*factor]。
    """
    # 確認 factor 為正整數
    assert isinstance(factor, int) and factor >= 1
    # 若沒給 kernel，預設用 [1]*factor，代表 nearest neighbor 上採樣
    if k is None:
        k = [1] * factor
    # k 轉為 numpy kernel，並做正規化，使常數輸入時信號能被正確縮放
    # k = _setup_kernel(k)：轉成二維、正規化，使 kernel sum=1
    # 再乘以 gain * (factor**2)，確保能量不會放大或縮小
    k = _setup_kernel(k) * (gain * (factor**2))
    # 計算 padding 大小，使上採樣後邊界像素對齊
    p = k.shape[0] - factor
    # 用 upfirdn2d 做 upsampling + FIR 濾波 + padding
    # pad=((p+1)//2 + factor - 1, p//2) 這樣設計能讓上採樣後 output shape 對齊
    return upfirdn2d(
        x,
        torch.tensor(k, device=x.device),
        up=factor,
        pad=((p + 1) // 2 + factor - 1, p // 2),
    )


def downsample_2d(x, k=None, factor=2, gain=1):
    r"""用指定的濾波器對一批二維影像進行下採樣.

    Args:
        x:      輸入的 4D tensor，shape 可為 [N, C, H, W] 或 [N, H, W, C]，
                分別是 batch, channel, 高度, 寬度。
        k:      FIR 濾波器（filter kernel），一維或二維都可。預設 [1]*factor，相當於 average pooling。
        factor: 下採樣倍率（預設 2，表示高/寬都除以 2）。
        gain:   輸出強度縮放係數（預設 1.0）。
    Returns:
        回傳下採樣後的 tensor，shape 是 [N, C, H // factor, W // factor]。
    """

    # 確認 factor 是正整數
    assert isinstance(factor, int) and factor >= 1

    # 如果沒指定 kernel，預設用 [1, 1]（factor=2 時），等於 average pooling
    if k is None:
        k = [1] * factor

    # 把 kernel 正規化，讓總和為 1，再乘上 gain（強度係數）
    k = _setup_kernel(k) * gain

    # 根據 kernel 長度與 downsample 倍率，計算 padding 大小
    p = k.shape[0] - factor

    # 用 upfirdn2d 做：padding → FIR 濾波（平滑）→ 下採樣（取樣）
    # pad=((p + 1) // 2, p // 2)：保證 output shape 正確，平衡左右/上下邊界
    return upfirdn2d(
        x, torch.tensor(k, device=x.device), down=factor, pad=((p + 1) // 2, p // 2)
    )
