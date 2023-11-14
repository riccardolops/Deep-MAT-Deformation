import lightning.pytorch as pl
from monai.transforms import ( Compose, LoadImaged, ToTensord, Spacingd, ScaleIntensityd, Resized, EnsureChannelFirstd )
from config import Config
from model.deepMATdeform import LitVoxel2MAT
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
from datasets.HeartDataset import HeartDataset, LoadObjd
import torch


cfg = Config()
keys = ['image', 'label']
transform_train = Compose([
    LoadImaged(keys=keys),
    LoadObjd(keys=['surface_vtx']),
    EnsureChannelFirstd(keys=keys),
    Spacingd(keys=keys, pixdim=(cfg.spacing, cfg.spacing, cfg.spacing)),
    ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
    Resized(keys=keys, spatial_size=cfg.resize_shape),
    ToTensord(keys=keys)
 ])
train_dataset = HeartDataset(cfg, 'train', transform_train)
train_dataloader = train_dataset.get_loader()
val_dataset = HeartDataset(cfg, 'val', transform_train)
val_dataloader = val_dataset.get_loader()

callbacks = [ModelCheckpoint(monitor='val_loss', dirpath=cfg.save_path, filename='{epoch:02d}-{val_loss:.2f}')]
def train():
    wandb.init(project="Deep-MAT-Deformation")

    cfg.learning_rate = wandb.config.learning_rate
    cfg.batch_norm = wandb.config.batch_norm
    cfg.first_layer_channels = wandb.config.first_layer_channels
    cfg.graph_conv_layer_count = wandb.config.graph_conv_layer_count

    logger = WandbLogger(project="Deep-MAT-Deformation", save_dir=cfg.save_path)
    trainer = pl.Trainer(accelerator="gpu", devices=[1], profiler=cfg.profiler, log_every_n_steps=cfg.eval_every, logger=logger, callbacks=callbacks, max_epochs=cfg.numb_of_epochs, default_root_dir=cfg.save_path)
    trainer.fit(LitVoxel2MAT(cfg), train_dataloader, val_dataloader)
    torch.cuda.empty_cache()

wandb.agent('segment_/Deep-MAT-Deformation/b0gsb4rf',function=train, project="Deep-MAT-Deformation")