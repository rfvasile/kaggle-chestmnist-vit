"""Defaults shared by every experiment. A `cfg_*.py` deep-copies this and overrides."""

from types import SimpleNamespace

COMPETITION = "22-aie-437-vision-transformer-from-scratch-on-chest-mnist"

cfg = SimpleNamespace()

# Modules, as import paths; the runners' docstrings list what each exposes
cfg.competition = COMPETITION
cfg.dataset = None
cfg.post_process_pipeline = None
cfg.metric = None
cfg.model = None

# Data
cfg.raw_dir = f"input/{COMPETITION}"
cfg.datamount = "datamount"
cfg.labels = [
    "atelectasis",
    "cardiomegaly",
    "effusion",
    "infiltration",
    "mass",
    "nodule",
    "pneumonia",
    "pneumothorax",
    "consolidation",
    "edema",
    "emphysema",
    "fibrosis",
    "pleural",
    "hernia",
]

cfg.n_folds = 5
cfg.rebuild = False

# Split seed, apart from cfg.seed so seed replicas share one split
cfg.fold_seed = 3

# AutoGluon (train_gluon.py)
cfg.problem_type = "binary"
cfg.eval_metric = "log_loss"
cfg.positive_class = 1  # None scores predict() instead of predict_proba()
cfg.presets = "medium"
cfg.hyperparameters = None
cfg.time_limit = None
cfg.ignored_columns = []
cfg.num_gpus = 0
cfg.refit_full = False
cfg.row_id_cols = []

# Torch (train_nn.py)
cfg.folds = [0, 1, 2, 3, 4]
cfg.epochs = 3
cfg.batch_size = 32
cfg.lr = 3e-5
cfg.weight_decay = 0.01
cfg.schedule = "linear"
cfg.warmup = 0.1  # in epochs
cfg.grad_accumulation = 1
cfg.clip_grad = 0.0
cfg.track_grad_norm = False
cfg.mixed_precision = True
cfg.amp_dtype = "float16"
cfg.num_workers = 2
cfg.pin_memory = True
cfg.eval_epochs = 1
cfg.eval_steps = None
cfg.keep_best = False
cfg.swa_start_epoch = None
cfg.swa_steps = None
cfg.save_checkpoint = False

# Logging
cfg.notes = ""
cfg.wandb_project = f"kaggle-{COMPETITION}"
cfg.wandb_mode = "online"

# Run
cfg.name = None
cfg.output_dir = None
cfg.seed = 3  # -1 draws a random one and records it
cfg.selection_metric = "auc"  # a key of calc_metric's dict
cfg.keep_predictor = False
cfg.test = True
