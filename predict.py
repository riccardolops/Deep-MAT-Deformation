from model.deepMATdeform import LitVoxel2MAT
from config import Config
import pytorch_lightning as pl
from Heart import Heart
import torchio as tio
import pyvista as pv
import numpy as np
from model.deepMATdeform import first_moment


cfg = Config()
preprocessing = tio.RescaleIntensity((0, 1))
transform_train = tio.Compose([
    tio.Resample(1.25),
    tio.CropOrPad(cfg.resize_shape, padding_mode=0),
    preprocessing,
])
train_dataset = Heart(cfg.dataset_path, 'train', transform_train)
train_dataloader = train_dataset.get_loader(cfg)
model = LitVoxel2MAT(cfg)
model = model.load_from_checkpoint("/home/rick/Documenti/Projects/models_waights/epoch=48-val_sample=38.55.ckpt", config=cfg)
trainer = pl.Trainer(accelerator="gpu", devices=[0], profiler=cfg.profiler, log_every_n_steps=cfg.eval_every, max_epochs=cfg.numb_of_epochs, default_root_dir=cfg.save_path)
predictions = trainer.predict(model, train_dataloader)

faces_padded = []
for face in model.mat_faces.squeeze().numpy():
    num_points = len(face)
    faces_padded.extend([num_points, *face])
faces_padded = np.array(faces_padded, dtype=np.int32)

lines_padded = []
for line in model.mat_lines.squeeze().numpy():
    num_points = len(line)
    lines_padded.extend([num_points, *line])
lines_padded = np.array(lines_padded, dtype=np.int32)
skl_mesh = pv.PolyData(model.mat_features.squeeze().numpy()[:, :-1], faces=faces_padded,
                            lines=lines_padded)

for patient in predictions:
    mask = patient[0]
    MAT = patient[1]
    target = patient[2]
    target_surface = target.squeeze().numpy()
    skl_meshp = skl_mesh.copy()
    vtx = MAT.squeeze().numpy()
    skl_meshp.points = vtx[:, :-1]
    centroid = first_moment(mask)

    plot = pv.Plotter(notebook=False)
    plot.add_text("Target", font_size=30, color='#FF0000', position='upper_right')
    plot.add_points(target_surface, color='#FF0000', opacity=0.20)

    plot.add_text("X", font_size=30, color='#00FF00')
    plot.add_mesh(skl_meshp, color='#00FF00')
    for point, rad in zip(skl_meshp.points, vtx[:, -1]):
        plot.add_mesh(pv.Sphere(center=point, radius=rad), opacity=0.20)
    # plotter.add_points(pred_points.detach().cpu().numpy())
    plot.show()

print(predictions)

