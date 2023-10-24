from Aorta import Aorta
import torchio as tio
from config import Config
import open3d as o3d
from skimage import measure

cfg = Config()
transform = tio.Compose([
    tio.ToCanonical(),
    tio.CropOrPad((512, 512, 512), padding_mode=0),
    tio.Resample(3),
    tio.CropOrPad(cfg.resize_shape, padding_mode=0),
    tio.RescaleIntensity((0, 1)),
])
dataset = Aorta(cfg.dataset_path, transform)
dataloader = dataset.get_loader(cfg)

for batch in dataloader:
    segmentations = batch['segmentation']  # Binary segmentations
    binary_segmentation = segmentations['data']
    binary_segmentation_np = binary_segmentation.squeeze().cpu().numpy()
    mcs_vert, mcs_tri, _, _ = measure.marching_cubes(binary_segmentation_np, 0)
    mcs_mesh = o3d.geometry.TriangleMesh()
    mcs_mesh.vertices = o3d.utility.Vector3dVector(mcs_vert)
    mcs_mesh.triangles = o3d.utility.Vector3iVector(mcs_tri)

    # Save the TriangleMesh as an OBJ file
    o3d.io.write_triangle_mesh(batch['path'][0] + '.obj', mcs_mesh)