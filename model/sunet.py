import torch
import torch.nn as nn
import torch.nn.functional as F
from model.flop_counter import count_flops_forward


class REBNConv(nn.Module):
    """
    Conv -> BN -> ReLU block.

    :param in_ch: Number of input channels.
    :param out_ch: Number of output channels.
    :param k: Convolution kernel size.
    :param dilation: Dilation rate.
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, dilation: int = 1) -> None:
        super().__init__()
        p = dilation * (k // 2)
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=p, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class RSU4FBlock(nn.Module):
    """
    RSU-4F micro U-block without pooling, using dilations for context.

    Structure (all mid-width):
        in -> conv1 ->
            conv2(dil=2) ->
                conv3(dil=4) ->
                    conv4(dil=8) ->
                dec3(dil=4) cat conv3 ->
            dec2(dil=2) cat conv2 ->
        dec1(dil=1) cat conv1 -> out + in_proj

    :param in_ch: Number of input channels.
    :param mid_ch: Hidden channels within the block.
    :param out_ch: Output channels of the block.
    """

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int) -> None:
        super().__init__()
        self.in_proj = REBNConv(in_ch, out_ch, k=3, dilation=1)

        self.c1 = REBNConv(out_ch, mid_ch, k=3, dilation=1)
        self.c2 = REBNConv(mid_ch, mid_ch, k=3, dilation=2)
        self.c3 = REBNConv(mid_ch, mid_ch, k=3, dilation=4)
        self.c4 = REBNConv(mid_ch, mid_ch, k=3, dilation=8)

        self.d3 = REBNConv(mid_ch * 2, mid_ch, k=3, dilation=4)
        self.d2 = REBNConv(mid_ch * 2, mid_ch, k=3, dilation=2)
        self.d1 = REBNConv(mid_ch * 2, out_ch, k=3, dilation=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.in_proj(x)

        x1 = self.c1(x_in)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x4 = self.c4(x3)

        y3 = self.d3(torch.cat([x4, x3], dim=1))
        y2 = self.d2(torch.cat([y3, x2], dim=1))
        y1 = self.d1(torch.cat([y2, x1], dim=1))

        return y1 + x_in


class StraightU2Net(nn.Module):
    """Single-stream U2-Net-style network for segmentation.

    This is a straight stack of RSU4F blocks, no encoder-decoder, no multi-scale side outputs.
    Spatial size is preserved. Useful when you want U2-Net's micro-U context without the big U.

    :param in_ch: Number of input channels.
    :param out_ch: Number of output channels.
    :param base_ch: Base channel width for the first projection and RSU blocks.
    :param mid_ch: Mid-channel width inside each RSU4F block.
    :param num_blocks: Number of RSU4F blocks stacked in sequence.
    :param dropout: Dropout probability after each block output. Set 0 to disable.
    """

    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 1,
        base_ch: int = 64,
        mid_ch: int = 32,
        num_blocks: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")

        mean_t = torch.tensor([0.69662594, 0.5239926, 0.44156789], dtype=torch.float32).view(1, -1, 1, 1)
        std_t = torch.tensor([0.17165191, 0.18755298, 0.1873925], dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("mean", mean_t)
        self.register_buffer("std", std_t)

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )

        blocks = []
        for _ in range(num_blocks):
            blocks.append(RSU4FBlock(base_ch, mid_ch, base_ch))
            if dropout > 0.0:
                blocks.append(nn.Dropout2d(p=dropout))
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Conv2d(base_ch, out_ch, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :returns: Segmentation logits in shape (N, out_ch, H, W).
        """
        feats = self.stem((x - self.mean) / self.std)
        feats = self.blocks(feats)
        logits = self.head(feats)
        return F.sigmoid(logits)


# ---- quick sanity check ----
if __name__ == "__main__":
    TEST_DIM = 320
    model = StraightU2Net(in_ch=3, out_ch=1, base_ch=32, mid_ch=16, num_blocks=2, dropout=0.05)
    x = torch.randn(2, 3, TEST_DIM, TEST_DIM)
    y = model(x)
    print("out:", y.shape, f"\tparams: {sum(p.numel() for p in model.parameters()) :,}")

    import time

    test_tensor = torch.rand(1, 3, TEST_DIM, TEST_DIM)
    flops = count_flops_forward(model, test_tensor)
    print(f"{flops = :,}")

    start = time.perf_counter()
    for _ in range(10):
        model(test_tensor)
    end = time.perf_counter()
    print(f"Time taken: {end - start:0.4f} seconds\n"
          f"{(end - start) / 10:0.4f} iterations per second")
