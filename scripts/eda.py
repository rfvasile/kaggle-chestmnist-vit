"""Exploratory analysis of the ChestMNIST data; docs/data.md says why each cell is here.

python scripts/eda.py    # or cell by cell in the REPL
"""

# %%
from copy import deepcopy
from importlib import import_module

import matplotlib.pyplot as plt

# %%
# Data
cfg = deepcopy(import_module("configs.cfg_b1").cfg)
ds = import_module(cfg.dataset)
ds.build(cfg)
train = ds.load(cfg, "train")
train_images = ds.images(cfg, "train")
labels = train[cfg.labels]
print(f"{len(train)} train images of {train_images.shape[1:]} px")

# %%
# Label prevalence
rates = labels.mean()
rates["no finding"] = labels.sum(axis=1).eq(0).mean()
rates.sort_values().plot.barh(figsize=(6, 5), title="share of train images")
plt.show()
print(rates.sort_values(ascending=False).round(4).to_string())

# %%
# Random train images
sample = train.sample(16, random_state=cfg.seed)
fig, axes = plt.subplots(2, 8, figsize=(12, 3.5))
for ax, (_, row) in zip(axes.flat, sample.iterrows()):
    ax.imshow(train_images[row["img_idx"]], cmap="gray")
    ax.set_title(", ".join(row[cfg.labels].index[row[cfg.labels] == 1]) or "no finding", fontsize=6)
    ax.axis("off")
plt.show()
