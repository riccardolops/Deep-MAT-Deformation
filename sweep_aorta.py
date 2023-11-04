import pytorch_lightning as pl
from monai.transforms import ( Compose, LoadImaged, ToTensord, Spacingd, ScaleIntensityd, Resized, EnsureChannelFirstd )
from config import Config
from model.deepMATdeform import LitVoxel2MAT
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from datasets.AortaDataset import AortaDataset, LoadObjd


cfg = Config()
keys = ['image', 'label']
transform_train = Compose([
    LoadImaged(keys=keys),
    LoadObjd(keys=['surface_vtx']),
    EnsureChannelFirstd(keys=keys),
    Spacingd(keys=keys, pixdim=(3, 3, 3)),
    ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
    Resized(keys=keys, spatial_size=cfg.resize_shape),
    ToTensord(keys=keys)
 ])
train_dataset = AortaDataset(cfg, 'train', transform_train)
train_dataloader = train_dataset.get_loader()
val_dataset = AortaDataset(cfg, 'val', transform_train)
val_dataloader = val_dataset.get_loader()

callbacks = [ModelCheckpoint(monitor='val_sample', dirpath=cfg.save_path, filename='{epoch:02d}-{val_sample:.2f}')]
def train():
    wandb.init(project="Deep-MAT-Deformation")

    cfg.learning_rate = wandb.config.learning_rate
    cfg.lambda_p2s = wandb.config.lambda_p2s
    cfg.lambda_radius = wandb.config.lambda_radius
    cfg.lambda_ce = wandb.config.lambda_ce
    cfg.lambda_dice = wandb.config.lambda_dice

    logger = WandbLogger(project="Deep-MAT-Deformation", save_dir="\\Users\\rick\\Documents\\Projects\\DMD_wandb")
    trainer = pl.Trainer(accelerator="gpu", profiler=cfg.profiler, log_every_n_steps=cfg.eval_every, logger=logger, callbacks=callbacks, max_epochs=cfg.numb_of_epochs, default_root_dir=cfg.save_path)
    trainer.fit(LitVoxel2MAT(cfg), train_dataloader, val_dataloader)

wandb.agent('segment_/Deep-MAT-Deformation/de69c4ru',function=train, project="Deep-MAT-Deformation")