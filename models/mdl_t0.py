"""t0 model: an MLP over the feature vector, BCE on one logit."""

from torch import nn

CFG_KEYS = {"hidden", "dropout"}


class Net(nn.Module):
    def __init__(self, dataset, cfg, mode="train"):
        super().__init__()
        n_features = dataset.features.shape[1]
        self.mlp = nn.Sequential(
            nn.Linear(n_features, cfg.hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, 1),
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, batch):
        # Logit, and the BCE loss when labelled
        logits = self.mlp(batch["features"]).squeeze(-1).float()
        out = {"logits": logits}
        if "target" in batch:
            out["loss"] = self.loss_fn(logits, batch["target"])
        return out
