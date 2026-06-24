import torch
from torch import nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from model.flop_counter import count_flops_forward

class DeepLabV3MobileNetV3(nn.Module):
    """
    DeepLabV3 with MobileNetV3 backbone

    Parameters
    ----------
    num_classes : int
        Output channels. Use 1 for a single foreground/background mask.
    """

    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()
        self.net = deeplabv3_mobilenet_v3_large(
            weights=None,             # classifier head uninitialized
            weights_backbone=None,    # backbone uninitialized
            num_classes=num_classes,  # 1-channel logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)["out"]  # [N, C, H, W], resized to input size
        return torch.sigmoid(y)


if __name__ == '__main__':
    # ---- quick sanity check ----
    TEST_DIM = 256
    model = DeepLabV3MobileNetV3()
    x = torch.randn(2, 3, TEST_DIM, TEST_DIM)
    y = model(x)
    print(f"params: {sum(p.numel() for p in model.parameters()) :,}")
    test_tensor = torch.rand(1, 3, TEST_DIM, TEST_DIM)
    flops = count_flops_forward(model, test_tensor)
    print(f"{flops = :,}")
