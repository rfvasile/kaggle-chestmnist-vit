# %%
"""EDA for docs/etapes-pipeline.md section 3: one helper per question, all run by explore()."""

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from data import prepare
from data.prepare import ID_COL

SPLITS = ("train", "test")


# Images
def png_formats(cfg, split: str) -> pd.DataFrame:
    """Width, height and PIL mode of every raw PNG in split."""
    rows = []
    for i in prepare.load(cfg, split)[ID_COL]:
        with Image.open(prepare.raw_image_dir(cfg, split) / f"{i}.png") as im:
            rows.append((*im.size, im.mode))
    return pd.DataFrame(rows, columns=["width", "height", "mode"])


def flat_chunks(imgs: np.ndarray, size: int = 8192):
    """(n, H*W) slices of imgs, so float copies stay small."""
    for start in range(0, len(imgs), size):
        yield imgs[start : start + size].reshape(-1, imgs[0].size)


def pixel_histogram(imgs: np.ndarray) -> np.ndarray:
    """Count of each gray level 0-255 over all pixels."""
    return np.sum([np.bincount(c.ravel(), minlength=256) for c in flat_chunks(imgs)], axis=0)


def pixel_stats(imgs: np.ndarray) -> pd.DataFrame:
    """Per-image mean, std, min and max gray level, indexed like imgs."""
    parts = []
    for c in flat_chunks(imgs):
        x = c.astype(np.float32)
        parts.append(np.column_stack([x.mean(1), x.std(1), x.min(1), x.max(1)]))
    return pd.DataFrame(np.concatenate(parts), columns=["mean", "std", "min", "max"])


def duplicates(splits: dict[str, np.ndarray]) -> pd.DataFrame:
    """Images whose pixels exactly match another's, within or across splits."""
    rows = [(s, i, hashlib.md5(img.tobytes()).hexdigest()) for s, imgs in splits.items() for i, img in enumerate(imgs)]
    df = pd.DataFrame(rows, columns=["split", "img_idx", "hash"])
    return df[df.duplicated("hash", keep=False)].sort_values(["hash", "split"], ignore_index=True)


def outliers(stats: pd.DataFrame, n: int = 8) -> dict[str, pd.Index]:
    """img_idx of the n most extreme images by contrast and brightness."""
    return {
        "lowest std": stats["std"].nsmallest(n).index,
        "highest std": stats["std"].nlargest(n).index,
        "darkest": stats["mean"].nsmallest(n).index,
        "brightest": stats["mean"].nlargest(n).index,
    }


# Labels
def label_stats(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Missing values, positives, prevalence and the all-negative predictor's accuracy, rarest first."""
    prevalence = df[labels].mean()
    return pd.DataFrame(
        {
            "missing": df[labels].isna().sum(),
            "positives": df[labels].sum(),
            "prevalence": prevalence,
            "all_negative_acc": 1 - prevalence,
        }
    ).sort_values("positives")


def labels_per_image(df: pd.DataFrame, labels: list[str]) -> pd.Series:
    """Number of images per count of positive labels."""
    return df[labels].sum(axis=1).value_counts().sort_index()


def fold_prevalence(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Prevalence of each label (rows) in each fold (columns)."""
    return df.groupby("fold")[labels].mean().T


def cooccurrence(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Label pairs: images with both, and their phi correlation, most frequent first."""
    y = df[labels].to_numpy()
    both, phi = y.T @ y, np.corrcoef(y, rowvar=False)
    i, j = np.triu_indices(len(labels), k=1)
    names = np.array(labels)
    pairs = pd.DataFrame({"a": names[i], "b": names[j], "both": both[i, j], "phi": phi[i, j]})
    return pairs.sort_values("both", ascending=False, ignore_index=True)


def label_pixel_diffs(df: pd.DataFrame, stats: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Mean brightness and contrast of each label's positives minus its negatives, in gray levels."""
    s = stats.loc[df["img_idx"], ["mean", "std"]].to_numpy()
    rows = {}
    for label in labels:
        pos = df[label].to_numpy() == 1
        rows[label] = s[pos].mean(axis=0) - s[~pos].mean(axis=0)
    diffs = pd.DataFrame.from_dict(rows, orient="index", columns=["brightness_diff", "contrast_diff"])
    return diffs.sort_values("brightness_diff", key=abs, ascending=False)


def label_examples(df: pd.DataFrame, labels: list[str], n: int = 8, seed: int = 0) -> dict[str, np.ndarray]:
    """img_idx of n random images with no label, then n per label."""
    groups = {"no finding": df[df[labels].sum(axis=1) == 0]} | {label: df[df[label] == 1] for label in labels}
    return {name: g["img_idx"].sample(n, random_state=seed).to_numpy() for name, g in groups.items()}


# Plots
def image_grid(imgs: np.ndarray, rows: dict) -> plt.Figure:
    """One row of images per key, titled by the key."""
    ncols = max(len(idx) for idx in rows.values())
    fig, axes = plt.subplots(len(rows), ncols, figsize=(ncols * 1.2, len(rows) * 1.4), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for r, (name, idx) in enumerate(rows.items()):
        axes[r, 0].set_title(name, loc="left", fontsize=9)
        for c, i in enumerate(idx):
            axes[r, c].imshow(imgs[i], cmap="gray", vmin=0, vmax=255)
    fig.tight_layout()
    return fig


def histogram_plot(hists: dict[str, np.ndarray]) -> plt.Figure:
    """Share of pixels at each gray level, one line per split."""
    fig, ax = plt.subplots(figsize=(8, 3))
    for split, h in hists.items():
        ax.plot(h / h.sum(), label=split)
    ax.set(xlabel="gray level", ylabel="share of pixels", yscale="log")
    ax.legend()
    fig.tight_layout()
    return fig


# Report
def heading(title: str) -> None:
    print(f"\n## {title}\n")


def explore(cfg, out_dir: str = "output/eda") -> None:
    """Print the facts behind each question of section 3, and save its figures to out_dir."""
    prepare.build(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train, test = (prepare.load(cfg, s) for s in SPLITS)
    imgs = {s: prepare.images(cfg, s) for s in SPLITS}
    stats = {s: pixel_stats(x) for s, x in imgs.items()}
    hists = {s: pixel_histogram(x) for s, x in imgs.items()}
    formats = pd.concat({s: png_formats(cfg, s) for s in SPLITS}, names=["split"]).droplevel(1)
    pd.set_option("display.width", 160)

    heading("Image size")
    print(formats.groupby("split")[["width", "height"]].value_counts())

    heading("Channels")
    print(formats.groupby("split")["mode"].value_counts())

    heading("Pixel range")
    for s, h in hists.items():
        used = np.flatnonzero(h)
        print(f"{s}: {used.min()}-{used.max()}, {256 - len(used)} unused gray levels")

    heading("Hidden preprocessing")
    for s, st in stats.items():
        h = hists[s]
        full = ((st["min"] == 0) & (st["max"] == 255)).mean()
        print(
            f"{s}: {full:.1%} of images span 0-255; {h[0] / h.sum():.2%} of pixels at 0, {h[255] / h.sum():.2%} at 255"
        )
    histogram_plot(hists).savefig(out / "pixel_histogram.png")

    heading("Label frequency and accuracy trap")
    print(label_stats(train, cfg.labels))

    heading("Fold proportions")
    print(fold_prevalence(train, cfg.labels).round(4))

    heading("Labels per image")
    print(labels_per_image(train, cfg.labels))

    heading("Co-occurrences")
    print(cooccurrence(train, cfg.labels).head(10))
    print(f"\nexamples per label: {out / 'labels.png'}")
    image_grid(imgs["train"], label_examples(train, cfg.labels)).savefig(out / "labels.png")

    heading("Labels vs brightness and contrast")
    print(label_pixel_diffs(train, stats["train"], cfg.labels).round(2))

    heading("Contamination")
    print(f"ids in both splits: {len(set(train[ID_COL]) & set(test[ID_COL]))}")
    print(f"train rows without a fold: {(train['fold'] < 0).sum()}")

    heading("Duplicates")
    dups = duplicates(imgs)
    print(f"{dups['hash'].nunique()} groups of identical images, {len(dups)} images")
    print(dups.groupby("hash")["split"].agg(lambda s: "+".join(sorted(set(s)))).value_counts())

    heading("Anomalies and visual quality")
    for s, st in stats.items():
        print(f"{s} extremes: {out / f'outliers_{s}.png'}")
        image_grid(imgs[s], outliers(st)).savefig(out / f"outliers_{s}.png")

    heading("Train vs test brightness and contrast")
    print(pd.concat(stats, names=["split"]).groupby("split")[["mean", "std"]].describe().T.round(2))
    plt.close("all")


# %%

if __name__ == "__main__":
    from importlib import import_module

    explore(import_module("configs.default_config").cfg)
