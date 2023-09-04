import lightning.pytorch as pl
from Heart import Heart
import torchio as tio
from config import Config
from model.voxel2mesh import LitVoxel2Mesh, Voxel2Mesh
from lightning.pytorch.loggers.wandb import WandbLogger
from model.callbacks import PyVistaGifCallback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint


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
    ckpt_path = '/home/rick/projects/DMD_waights/epoch=479-val_loss=70.86.ckpt'
else:
    ckpt_path = None

logger = WandbLogger(project="DMD", save_dir="./logs/")
callbacks = [PyVistaGifCallback(), ModelCheckpoint(monitor='val_loss', dirpath=cfg.save_path, filename='{epoch:02d}-{val_loss:.2f}')]
trainer = pl.Trainer(accelerator="gpu", devices=[0], profiler=cfg.profiler, log_every_n_steps=cfg.eval_every, logger=logger, callbacks=callbacks, max_epochs=cfg.numb_of_epochs, default_root_dir=cfg.save_path)
trainer.fit(LitVoxel2Mesh(cfg), train_dataloader, val_dataloader, ckpt_path=ckpt_path)
