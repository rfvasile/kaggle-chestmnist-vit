# Template #

## Setup ##

``` shell
docker compose --profile gpu up -d
export COMP=<competition-slug>
kaggle competitions download -c $COMP -p input/$COMP
bash -c "cd input/$COMP && unzip -o $COMP.zip && rm $COMP.zip"
```

## Run ##

``` shell
docker compose exec -T gpu python train_gluon.py -C cfg_b1
docker compose exec -T gpu python train_nn.py -C cfg_t0 --epochs 1
docker compose exec -T gpu python -m pytest tests/ -q
```

## Submit ##

``` shell
kaggle kernels init -p <notebook-dir>
kaggle kernels push -p <notebook-dir>
```
