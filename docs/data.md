# Data

## EDA

`scripts/eda.py`, one cell each.

| Cell | Shows | Decides |
|---|---|---|
| Label prevalence | Measures class imbalance. | Stratifying folds on every label; whether to weight the loss. |
| Random train images | Checks the images load and look right. | Which augmentations to use. |

## Findings

- 89,687 train / 22,433 test images, 64×64 grayscale, 14 labels.
- 54.1% of train images have no label.
- Positive rate: 17.8% (infiltration) down to 0.2% (hernia, 185 images).

## Published work

- ViTs trained from scratch lean on augmentation and regularization; with enough compute, these matched about 10× more data [1].
- Random horizontal flips work on chest X-rays: CheXNet used them on ChestX-ray14, the source of ChestMNIST [2].

## References

1. Steiner et al., *How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers*, 2021. https://arxiv.org/abs/2106.10270
2. Rajpurkar et al., *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning*, 2017. https://arxiv.org/abs/1711.05225
