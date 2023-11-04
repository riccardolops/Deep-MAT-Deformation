import pytorch_lightning as pl
from Heart import Heart
import torchio as tio
from config import Config
from model.deepMATdeform import LitVoxel2MAT
from pytorch_lightning.loggers.wandb import WandbLogger
from model.callbacks import GifCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint



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
    ckpt_path = '/home/rick/Documenti/Projects/models_waights/epoch=48-val_sample=38.55.ckpt'
else:
    ckpt_path = None

logger = WandbLogger(project="Deep-MAT-Deformation", save_dir="/home/rick/Documenti/Projects/DMD_wandb")
callbacks = [GifCallback(), ModelCheckpoint(monitor='val_sample', dirpath=cfg.save_path, filename='{epoch:02d}-{val_sample:.2f}')]
trainer = pl.Trainer(accelerator="gpu", devices=[1], profiler=cfg.profiler, log_every_n_steps=cfg.eval_every, logger=logger, callbacks=callbacks, max_epochs=cfg.numb_of_epochs, default_root_dir=cfg.save_path)
trainer.fit(LitVoxel2MAT(cfg), train_dataloader, val_dataloader, ckpt_path=ckpt_path)