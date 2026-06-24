import torch
from torch.utils.flop_counter import FlopCounterMode


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
