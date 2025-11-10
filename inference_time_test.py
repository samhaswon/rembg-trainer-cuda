import time
import torch
from torch import nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from model import U2NET, U2NETP


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
    ITERATIONS = 10
    torch.set_grad_enabled(False)

    # Prepare inputs
    test_256 = torch.randn(1, 3, 256, 256)
    test_320 = torch.randn(1, 3, 320, 320)
    test_512 = torch.randn(1, 3, 512, 512)
    test_1024 = torch.randn(1, 3, 1024, 1024)
    test_1280 = torch.randn(1, 3, 1280, 1280)
    test_1728 = torch.randn(1, 3, 1728, 1728)
    test_2048 = torch.randn(1, 3, 2048, 2048)

    # Instantiate models and put them in eval mode
    model = U2NET(3, 1).eval()
    model_p = U2NETP(3, 1).eval()
    model_dlmv = DeepLabV3MobileNetV3(num_classes=1).eval()

    inputs = {
        "256x256": test_256,
        "320x320": test_320,
        "512x512": test_512,
        "1024x1024": test_1024,
        "1280x1280": test_1280,
        "1728x1728": test_1728,
        "2048x2048": test_2048,
    }

    models = {
        "U2NET": model,
        "U2NETP": model_p,
        "DeepLabV3MobileNetV3": model_dlmv,
    }
    # print("\nCompiling models", end="")
    # for model_name, mdl in models.copy().items():
    #     models[model_name + " Compiled"] = torch.compile(mdl, mode="max-autotune-no-cudagraphs").eval()
    #     print(".", end="")
    # print()

    for model_name, mdl in models.items():
        print(f"\n=== {model_name} ===")
        # with torch.inference_mode():
        for size_name, inp in inputs.items():
            # Warm-up to avoid measuring lazy init cost
            _ = mdl(inp)

            start = time.perf_counter()
            for _ in range(ITERATIONS):
                _ = mdl(inp)
            end = time.perf_counter()

            avg_time = (end - start) / ITERATIONS
            print(f"{size_name}: {avg_time:.4f} seconds per inference")

    for model_name, mdl in models.items():
        print(f"\n=== {model_name} (`torch.compile`) ===")
        # with torch.inference_mode():
        for size_name, inp in inputs.items():
            # Warm-up to avoid measuring lazy init cost
            mdl_c = torch.compile(mdl, mode="max-autotune-no-cudagraphs").eval()
            _ = mdl_c(inp)

            start = time.perf_counter()
            for _ in range(ITERATIONS):
                _ = mdl_c(inp)
            end = time.perf_counter()

            avg_time = (end - start) / ITERATIONS
            print(f"{size_name}: {avg_time:.4f} seconds per inference")
