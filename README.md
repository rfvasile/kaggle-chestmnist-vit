# <title> #

## Setup ##

``` shell
uv sync
export COMP=<competition>
uv run kaggle competitions download -c $COMP -p input/$COMP
bash -c "cd input/$COMP && unzip -o $COMP.zip && rm $COMP.zip"
```

## Run ##

``` shell
uv run python train_gluon.py -C cfg_b1
uv run python train_nn.py -C cfg_t0 --epochs 1
uv run pytest -q
```

## Submit ##

``` shell
uv run kaggle kernels init -p <notebook-dir>
uv run kaggle kernels push -p <notebook-dir>
```
