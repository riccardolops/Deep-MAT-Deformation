from Heart import Heart
import torchio as tio
from config import Config
import open3d as o3d
from skimage import measure
from pathlib import Path


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

for batch in train_dataloader:
    segmentations = batch['segmentation']  # Binary segmentations
    binary_segmentation = segmentations['data']
    binary_segmentation_np = binary_segmentation.squeeze().cpu().numpy()
    mcs_vert, mcs_tri, _, _ = measure.marching_cubes(binary_segmentation_np, 0)
    mcs_mesh = o3d.geometry.TriangleMesh()
    mcs_mesh.vertices = o3d.utility.Vector3dVector(mcs_vert)
    mcs_mesh.triangles = o3d.utility.Vector3iVector(mcs_tri)

    # Save the TriangleMesh as an OBJ file
    o3d.io.write_triangle_mesh(str((Path(cfg.dataset_path) / batch['patient']['surface'][0])), mcs_mesh)

for batch in val_dataloader:
    segmentations = batch['segmentation']  # Binary segmentations
    binary_segmentation = segmentations['data']
    binary_segmentation_np = binary_segmentation.squeeze().cpu().numpy()
    mcs_vert, mcs_tri, _, _ = measure.marching_cubes(binary_segmentation_np, 0)
    mcs_mesh = o3d.geometry.TriangleMesh()
    mcs_mesh.vertices = o3d.utility.Vector3dVector(mcs_vert)
    mcs_mesh.triangles = o3d.utility.Vector3iVector(mcs_tri)

    # Save the TriangleMesh as an OBJ file
    o3d.io.write_triangle_mesh(str((Path(cfg.dataset_path) / batch['patient']['surface'][0])), mcs_mesh)


