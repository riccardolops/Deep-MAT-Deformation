import lightning.pytorch as pl
from Heart import Heart
from AortaDataset import AortaDataset, LoadObjd
import torchio as tio
from monai.transforms import ( Compose, LoadImaged, ToTensord, Spacingd, ScaleIntensityd, Resized, EnsureChannelFirstd)
from config import Config
from model.voxel2mesh import LitVoxel2Mesh
from lightning.pytorch.loggers.wandb import WandbLogger
from model.callbacks import PyVistaGifCallback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint


cfg = Config()
# preprocessing = tio.RescaleIntensity((0, 1))
# transform_train = tio.Compose([
#     tio.Resample(1.25),
#     tio.CropOrPad(cfg.resize_shape, padding_mode=0),
#     preprocessing,
# ])
# transform_val = tio.Compose([
#     tio.Resample(1.25),
#     tio.CropOrPad(cfg.resize_shape, padding_mode=0),
#     preprocessing,
# ])
# train_dataset = Heart(cfg.dataset_path, 'train', transform_train)
# train_dataloader = train_dataset.get_loader(cfg)
# val_dataset = Heart(cfg.dataset_path, 'val', transform_val)
# val_dataloader = val_dataset.get_loader(cfg)
keys = ['image', 'label']
transform_train = Compose([
    LoadImaged(keys=keys),
    LoadObjd(keys=['surface_vtx']),
    EnsureChannelFirstd(keys=keys),
    Spacingd(keys=keys, pixdim=(3, 3, 3)),
    ScaleIntensityd(keys=keys, minv=0.0, maxv=1.0),
    Resized(keys=keys, spatial_size=cfg.resize_shape),
    ToTensord(keys=keys)
 ])
train_dataset = AortaDataset(cfg, 'train', transform_train)
train_dataloader = train_dataset.get_loader()
val_dataset = AortaDataset(cfg, 'val', transform_train)
val_dataloader = val_dataset.get_loader()

if cfg.restore_ckpt:
    ckpt_path = '/home/rick/projects/DMD_weights/epoch=479-val_loss=70.86.ckpt'
else:
    ckpt_path = None

logger = WandbLogger(project="Deep-MAT-Deformation", save_dir="/home/rick/Documenti/Projects/DMD_wandb")
callbacks = [ModelCheckpoint(monitor='val_loss', dirpath=cfg.save_path, filename='{epoch:02d}-{val_loss:.2f}')]
trainer = pl.Trainer(accelerator="gpu", devices=[1], profiler=cfg.profiler, log_every_n_steps=cfg.eval_every, logger=logger, callbacks=callbacks, max_epochs=cfg.numb_of_epochs, default_root_dir=cfg.save_path)
trainer.fit(LitVoxel2Mesh(cfg), train_dataloader, val_dataloader, ckpt_path=ckpt_path)
