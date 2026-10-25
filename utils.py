import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.amp.autocast_mode import autocast
from tqdm import tqdm


def calc_grad_norm(parameters, norm_ord=2) -> torch.Tensor | None:
    """Total gradient norm across parameters; None if there are no grads or the norm is nan/inf."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return None
    total_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(g.detach(), ord=norm_ord) for g in grads]), ord=norm_ord
    )
    if total_norm.isnan() or total_norm.isinf():
        return None
    return total_norm


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def predict(
    model, loader, batch_to_device, device, mixed_precision: bool, amp_dtype: str = "float16"
) -> tuple[dict, float]:
    """Every per-row output of the model over `loader` plus `idx`, and the mean loss."""
    model.eval()
    data, losses = defaultdict(list), []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict", leave=False):
            batch = batch_to_device(batch, device)
            with autocast(device, enabled=mixed_precision, dtype=getattr(torch, amp_dtype)):
                output = model(batch)
            for key, val in output.items():
                if key == "loss":
                    losses.append(val.item())
                else:
                    data[key].append(val.detach().float().cpu())
            data["idx"].append(batch["idx"].cpu())
    return {k: torch.cat(v) for k, v in data.items()}, float(np.mean(losses)) if losses else float("nan")
