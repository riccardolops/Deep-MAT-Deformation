import pytorch_lightning as pl
from Heart import Heart
import torchio as tio
from config import Config
from model.deepMATdeform import LitVoxel2MAT
from pytorch_lightning.loggers.wandb import WandbLogger
from model.callbacks import PyVistaGifCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

cfg = Config()
preprocessing = tio.RescaleIntensity((0, 1))
transform_train = tio.Compose([
    tio.Resample(1.25),
    tio.CropOrPad(cfg.resize_shape, padding_mode=0),
    preprocessing,
])
transform_val = tio.Compose([
    tio.Resample(1.25),
    tio.CropOrPad(cfg.resize_shape, padding_mode=0),
    preprocessing,
])
train_dataset = Heart(cfg.dataset_path, 'train', transform_train)
train_dataloader = train_dataset.get_loader(cfg)
val_dataset = Heart(cfg.dataset_path, 'val', transform_val)
val_dataloader = val_dataset.get_loader(cfg)
if cfg.restore_ckpt:
    ckpt_path = '/home/rick/projects/DMD_waights/epoch=59-val_loss=221.17.ckpt'
else:
    ckpt_path = None

callbacks = [ModelCheckpoint(monitor='val_sample', dirpath=cfg.save_path, filename='{epoch:02d}-{val_sample:.2f}')]
def train():
    wandb.init(project="Deep-MAT-Deformation")

    cfg.learning_rate = wandb.config.learning_rate
    cfg.lambda_p2s = wandb.config.lambda_p2s
    cfg.lambda_radius = wandb.config.lambda_radius
    cfg.lambda_ce = wandb.config.lambda_ce
    cfg.lambda_dice = wandb.config.lambda_dice

    logger = WandbLogger(project="Deep-MAT-Deformation", save_dir="/home/rick/Documenti/Projects/DMD_wandb")
    trainer = pl.Trainer(accelerator="gpu", profiler=cfg.profiler, log_every_n_steps=cfg.eval_every, logger=logger, callbacks=callbacks, max_epochs=cfg.numb_of_epochs, default_root_dir=cfg.save_path)
    trainer.fit(LitVoxel2MAT(cfg), train_dataloader, val_dataloader, ckpt_path=ckpt_path)

wandb.agent('segment_/Deep-MAT-Deformation/1uj8xb71',function=train, project="Deep-MAT-Deformation")