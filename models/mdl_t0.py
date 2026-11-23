# %%
import torch
from torch import nn

CFG_KEYS = {"patch_size"}


class ViT(nn.Module):
    def __init__(self, dataset, cfg, mode="train"):
        super().__init__()
        self.patch_size = cfg.patch_size

    def generate_patches(self, batch: torch.Tensor):
        """Generate patches from the input batch. See https://arxiv.org/abs/2010.11929, section 3.1."""
        # Input image: (B, C, H, W)
        B, C, H, W = batch.shape
        P = self.patch_size
        H_grid = int(H / P)
        W_grid = int(W / P)

        N = int((H * W) / P**2)

        # Reshape yields: B, C, height grid, height patch, column grid, column patch
        # This yields: (B, C, H_grid, H_patch, W_grid, W_patch)
        patches = batch.reshape((B, C, H_grid, P, W_grid, P))
        patches = torch.permute(
            patches,
            (0, 2, 4, 1, 3, 5),  # (B, H_grid, W_grid, C, P, P)
        )
        patches = patches.reshape((B, N, P**2 * C))

        return patches

    def forward(self, batch):
        pass


if __name__ == "__main__":
    from types import SimpleNamespace

    img = torch.arange(96).reshape((2, 3, 4, 4))
    ViT_model = ViT(None, SimpleNamespace(patch_size=2))
    patches = ViT_model.generate_patches(img)
    print(patches[0])  # expected first row: [0,1,4,5,16,17,20,21,32,33,36,37]
