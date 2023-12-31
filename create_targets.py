import os
from glob import glob

from config import Config
import open3d as o3d
from skimage import measure
import mcubes
from monai.data import Dataset, DataLoader
from monai.transforms import ( Compose, LoadImaged, ToTensord, Spacingd, ScaleIntensityd, Resized, EnsureChannelFirstd)
from monai.utils import first
import matplotlib.pyplot as plt

cfg = Config()


images = sorted(glob(os.path.join(cfg.dataset_path, '*/*/*.nrrd')))
labels = sorted(glob(os.path.join(cfg.dataset_path, '*/*/*.seg.nrrd')))
images = [image for image in images if image not in labels]
files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(images, labels)]
transforms = Compose([
    LoadImaged(keys=['label']),
    EnsureChannelFirstd(keys=['label']),
    Spacingd(keys=['label'], pixdim=(3, 3, 3)),
    Resized(keys=['label'], spatial_size=cfg.resize_shape),
    ToTensord(keys=['label'])
])
dataset = Dataset(data = files ,transform = transforms)
dataloader = DataLoader(dataset, cfg.batch_size)

for batch in dataloader:
    binary_segmentation = batch['label']  # Binary segmentations
    binary_segmentation_np = binary_segmentation.squeeze().cpu().numpy()
    # mcs_vert, mcs_tri, _, _ = measure.marching_cubes(binary_segmentation_np, 1.0)
    mcs_vert, mcs_tri = mcubes.marching_cubes(binary_segmentation_np, 0)
    mcs_mesh = o3d.geometry.TriangleMesh()
    mcs_mesh.vertices = o3d.utility.Vector3dVector(mcs_vert)
    mcs_mesh.triangles = o3d.utility.Vector3iVector(mcs_tri)

    # Save the TriangleMesh as an OBJ file
    print(binary_segmentation.size())
    print('Saving: ' + binary_segmentation.meta['filename_or_obj'][0][:-9] + '.obj')
    o3d.io.write_triangle_mesh(binary_segmentation.meta['filename_or_obj'][0][:-9] + '.obj', mcs_mesh)