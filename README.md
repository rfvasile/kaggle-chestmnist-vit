# ChestMNIST using ViT

Competition: [Vision Transformer from Scratch on ChestMNIST](https://www.kaggle.com/competitions/22-aie-437-vision-transformer-from-scratch-on-chest-mnist).

## Competition Rules

- Task: 14 independent probabilities per 64×64 grayscale chest X-ray.
- Metric: mean column-wise ROC AUC; every label counts equally, however rare.
- Model: patch embedding, positional embedding and Transformer encoder written by hand. No `torchvision.models.vit_*`, `timm`, `transformers` or pretrained weights.
- Data: the provided train set only. No external labeled data, no NIH ChestX-ray14 or MedMNIST files.

## Stages ##

| Stage | Lives in |
|---|---|
| Dataset | `data/ds_b1.py` |
| EDA | `scripts/eda.py`, `docs/data.md` |
| Evaluation metric | `metrics/metric_b1.py` |
| Cross-validation | folds in `data/ds_b1.py` |

## Run

``` shell
docker compose exec -T gpu python train_gluon.py -C cfg_b1
docker compose exec -T gpu python train_nn.py -C cfg_t0 --epochs 1
docker compose exec -T gpu python -m pytest tests/ -q
```

## Submit

``` shell
kaggle kernels init -p <notebook-dir>
kaggle kernels push -p <notebook-dir>
```
