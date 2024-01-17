from config import Config
from monai.transforms import ( Compose, LoadImaged, ToTensord, Spacingd, ScaleIntensityd, Resized, EnsureChannelFirstd )
from monai.transforms import ResampleToMatch, LoadImage
from model.deepMATdeform import LitVoxel2MAT
from monai.data import Dataset, DataLoader
from monai.utils import first
from utils.file_handle import write_ma
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from monai.data import NibabelWriter
import os
import scipy.io
import numpy as np
import pyvista as pv
import open3d as o3d


cfg = Config()
keys = ['image', 'label']
transform = Compose([
    LoadImaged(keys=keys),
    EnsureChannelFirstd(keys=keys),
    Spacingd(keys=keys, pixdim=(1.25, 1.25, 1.25)),
    ScaleIntensityd(keys=keys, minv=0.0, maxv=1.0),
    Resized(keys=keys, spatial_size=cfg.resize_shape),
    ToTensord(keys=keys)
 ])
image_name = "\\Users\\rick\\Documents\\datasets\\Task02_Heart\\imagesTr\\la_003.nii.gz"
label_name = "\\Users\\rick\\Documents\\datasets\\Task02_Heart\\labelsTr\\la_003.nii.gz"
target_pointcloud = "\\Users\\rick\\Documents\\datasets\\Task02_Heart\\labelsTr\\la_003_surf.obj"
subject_dict = [{"image": image_name, "label": label_name}]
dataset = Dataset(subject_dict, transform)
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=0, pin_memory=True)

model = LitVoxel2MAT.load_from_checkpoint("\\Users\\rick\\Documents\\Projects\\models_waights_heart\\epoch=76-val_loss=170.15.ckpt", config=cfg)
model.eval()
x = first(dataloader)
voxel_pred, MAT_deformed = model(x['image'].to(model.device))
mask = (torch.argmax(F.softmax(voxel_pred, dim=1), dim=1))

resample = False
interpolation="trilinear"

if resample:
    resampler = ResampleToMatch(mode=interpolation)
    loader = LoadImage(image_only=True, ensure_channel_first=True)
    _x = loader(os.path.abspath(x['image'].meta["filename_or_obj"][0]))
    #_prediction = resampler(mask, _x)
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.cos(x)
    scipy.io.savemat('test.mat', dict(x=x, y=y))

original_image = x['image'].squeeze()
expected_mask = x['label'].squeeze()
evaluated_mask = mask.squeeze(0).cpu()

def multi_slice_viewer(volume):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    ax0.volume = volume[0]
    ax1.volume = volume[1]
    ax2.volume = volume[2]
    ax0.index = volume[0].shape[0] // 2
    ax1.index = volume[1].shape[0] // 2
    ax2.index = volume[2].shape[0] // 2
    ax0.title.set_text('Original')
    ax1.title.set_text('Target')
    ax2.title.set_text('Output')
    ax0.imshow(volume[0][45], cmap='gray')
    ax1.imshow(volume[1][45])
    ax2.imshow(volume[2][45])
    fig.canvas.mpl_connect('scroll_event', process_key)

def process_key(event):
    fig = event.canvas.figure
    ax0 = fig.axes[0]
    ax1 = fig.axes[1]
    ax2 = fig.axes[2]
    if event.button == 'up':
        previous_slice(ax0)
        previous_slice(ax1)
        previous_slice(ax2)
    elif event.button == 'down':
        next_slice(ax0)
        next_slice(ax1)
        next_slice(ax2)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])

multi_slice_viewer([original_image, expected_mask, evaluated_mask])

#writer = NibabelWriter()
#writer.set_data_array(evaluated_mask)
#writer.set_metadata(_x.meta, resample=True)
#writer.write("test.nii.gz", verbose=True)

vertices = MAT_deformed[-1]
skel_xyzr = vertices.detach()
skel_xyz = skel_xyzr[:, :, :-1].cpu().squeeze()
skel_radius = skel_xyzr[:, :, -1].cpu().squeeze()

mat_edges = model.mat_edges.cpu().squeeze()
mat_faces = model.mat_faces.cpu().squeeze()

faces_padded = []
for face in model.mat_faces.cpu().squeeze().numpy():
    num_points = len(face)
    faces_padded.extend([num_points, *face])
faces_padded = np.array(faces_padded, dtype=np.int32)

lines_padded = []
for line in model.mat_lines.cpu().squeeze().numpy():
    num_points = len(line)
    lines_padded.extend([num_points, *line])
lines_padded = np.array(lines_padded, dtype=np.int32)

skl_mesh = pv.PolyData(model.mat_features.cpu().squeeze().numpy()[:, :-1], faces=faces_padded, lines=lines_padded)
skl_meshp = skl_mesh.copy()

plot = pv.Plotter()

#plot.add_text("Target", font_size=30, color='#FF0000', position='upper_right')
plot.add_text("Original MAT", font_size=30, color='#FF0000', position='upper_right')
plot.add_mesh(skl_meshp, color='#FF0000')
for point, rad in zip(skl_meshp.points, skel_radius.numpy()):
    plot.add_mesh(pv.Sphere(center=point, radius=rad), opacity=0.20)
    # plotter.add_points(pred_points.detach().cpu().numpy())

mesh = o3d.io.read_triangle_mesh(target_pointcloud)
verts = np.asarray(mesh.vertices)

#plot.add_points(verts, color='#FF0000', opacity=0.20)

plot.add_text("X", font_size=30, color='#00FF00')

skl_meshp.points = skel_xyz.numpy()
plot.add_mesh(skl_meshp, color='#00FF00')
for point, rad in zip(skl_meshp.points, skel_radius.numpy()):
    plot.add_mesh(pv.Sphere(center=point, radius=rad), opacity=0.20)
    # plotter.add_points(pred_points.detach().cpu().numpy())
plot.show()
write_ma('test.ma', skel_xyz, skel_radius, mat_edges, mat_faces)
