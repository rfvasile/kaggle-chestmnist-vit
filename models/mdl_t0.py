# %%
import numpy as np
import torch
from torch import nn

CFG_KEYS = {"hidden_size", "patch_size", "num_channels", "img_width", "img_height"}


def softmax(X: torch.Tensor):
    exp = X - X.max(dim=-1, keepdim=True)[0]
    return torch.exp(exp) / torch.exp(exp).sum(dim=-1, keepdim=True)


class MSA(nn.Module):
    def __init__(self, out_dim, in_dim, num_heads):
        super().__init__()
        self.W_k = torch.nn.Linear(in_dim, out_dim)
        self.W_v = torch.nn.Linear(in_dim, out_dim)
        self.W_q = torch.nn.Linear(in_dim, out_dim)
        self.W_o = torch.nn.Linear(out_dim, out_dim)

        self.num_heads = num_heads
        self.out_dim = out_dim
        self.in_dim = in_dim

        assert out_dim % num_heads == 0
        self.feats_per_head = int(out_dim / num_heads)

    def get_kvq(self, X):
        B, L, _ = X.shape
        assert X.shape == (B, L, self.in_dim)

        K = self.W_k(X)
        Q = self.W_q(X)
        V = self.W_v(X)
        assert K.shape == (B, L, self.out_dim)

        return K, Q, V

    def self_attention(self, K, Q, V):
        B = Q.shape[0]

        scores = torch.divide((Q @ K.mT), np.sqrt(K.shape[-1]))
        assert scores.shape == (B, self.num_heads, self.L, self.L)

        probs = softmax(scores)

        out = probs @ V
        assert V.shape == (B, self.num_heads, self.L, self.feats_per_head)
        assert out.shape == (B, self.num_heads, self.L, self.feats_per_head)

        return out

    def split_head(self, K: torch.Tensor):
        (B, L, _) = K.shape
        assert K.shape == (B, L, self.out_dim)

        out = K.swapaxes(-1, -2)
        out = out.reshape((B, self.num_heads, self.feats_per_head, L))
        out = out.swapaxes(-1, -2)

        assert out.shape == (B, self.num_heads, L, self.feats_per_head)
        return out

    def forward(self, batch):
        B, self.L, E = batch.shape
        assert (B, self.L, E) == (B, self.L, self.in_dim), (B, self.L, E, B, self.L, self.in_dim)
        K, Q, V = self.get_kvq(batch)

        # Split into multiple heads
        K = self.split_head(K)
        V = self.split_head(V)
        Q = self.split_head(Q)

        # Calculate attention scores
        attn_scores = self.self_attention(K, Q, V)

        # Convert to expected output
        assert attn_scores.shape == (B, self.num_heads, self.L, self.feats_per_head)
        attn_scores = attn_scores.swapaxes(1, 2)
        out = attn_scores.reshape((B, self.L, self.out_dim))

        # This final projection learns relationships between head ouputs
        out = self.W_o(out)
        return out


B, L, E = 2, 4, 9
batch = torch.arange(B * L * E, dtype=torch.float)
batch = batch.reshape((B, L, E))
net = MSA(out_dim=32, in_dim=E, num_heads=8)
net.forward(batch)

# %%


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

        self.msa = MSA(in_dim=self.patch_dim, out_dim=self.hidden_size)

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
