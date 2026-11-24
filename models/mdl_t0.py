# %%
import torch
from torch import nn

CFG_KEYS = {"hidden_size", "patch_size", "num_channels", "img_width", "img_height"}


class ViT(nn.Module):
    def __init__(self, cfg, mode="train"):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.hidden_size = cfg.hidden_size
        self.num_channels = cfg.num_channels

        self.patch_dim = self.patch_size**2 * self.num_channels
        self.num_patches = int((cfg.img_width * cfg.img_height) / self.patch_size**2)

        self.proj1 = nn.Linear(self.patch_dim, self.hidden_size)
        self.cls = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        self.pos_embeds = nn.Parameter(torch.randn(1, (self.num_patches + 1), self.hidden_size))

    def generate_patches(self, batch: torch.Tensor):
        """Generate patches from the input batch. See https://arxiv.org/abs/2010.11929, section 3.1."""
        # Input image: (B, C, H, W)
        B, C, H, W = batch.shape
        P = self.patch_size
        H_grid = int(H / P)
        W_grid = int(W / P)

        N = int((H * W) / P**2)

        # Reshape yields: (B, C, H_grid, H_patch, W_grid, W_patch)
        patches = batch.reshape((B, C, H_grid, P, W_grid, P))
        patches = torch.permute(
            patches,
            (0, 2, 4, 1, 3, 5),  # (B, H_grid, W_grid, C, P, P)
        )
        patches = patches.reshape((B, N, self.patch_dim))  # (B, N, C*P*P)

        return patches.to(torch.float)

    def forward(self, batch):
        B, _, _, _ = batch.shape

        # Generated patches (batch, num_patches, num_elements_per_patch)
        patches: torch.Tensor = self.generate_patches(batch)
        assert patches.shape == (B, self.num_patches, self.patch_dim)

        embeddings = self.proj1(patches)
        assert embeddings.shape == (B, self.num_patches, self.hidden_size)

        # Add class token
        cls_tokens = self.cls.expand((B, 1, self.hidden_size))
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        assert embeddings.shape == (B, self.num_patches + 1, self.hidden_size)

        # Add positional embeddings
        out = embeddings + self.pos_embeds

        return out


if __name__ == "__main__":
    from types import SimpleNamespace

    img = torch.arange(2 * 3 * 64 * 64).reshape((2, 3, 64, 64))
    ViT_model = ViT(SimpleNamespace(patch_size=8, num_channels=3, hidden_size=10, img_width=64, img_height=64))
    patches = ViT_model.forward(img)
    print(patches[0])
