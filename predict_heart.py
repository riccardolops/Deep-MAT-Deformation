from config import Config
from monai.transforms import ( Compose, LoadImaged, ToTensord, Spacingd, ScaleIntensityd, Resized, EnsureChannelFirstd )
from model.deepMATdeform import LitVoxel2MAT
from monai.data import Dataset, DataLoader
from monai.utils import first
from utils.file_handle import write_ma


cfg = Config()
keys = ['image']
transform = Compose([
    LoadImaged(keys=keys),
    EnsureChannelFirstd(keys=keys),
    Spacingd(keys=keys, pixdim=(1.25, 1.25, 1.25)),
    ScaleIntensityd(keys=keys, minv=0.0, maxv=1.0),
    Resized(keys=keys, spatial_size=cfg.resize_shape),
    ToTensord(keys=keys)
 ])
image_name = "\\Users\\rick\\Documents\\datasets\\Task02_Heart\\imagesTr\\la_003.nii.gz"
subject_dict = [{"image": image_name}]
dataset = Dataset(subject_dict, transform)
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=0, pin_memory=True)

model = LitVoxel2MAT.load_from_checkpoint("\\Users\\rick\\Documents\\Projects\\models_waights_heart\\epoch=89-val_loss=145.98.ckpt", config=cfg)
model.eval()
x = first(dataloader)
voxel_pred, MAT_deformed = model(x['image'].to(model.device))
vertices = MAT_deformed[-1]
skel_xyzr = vertices
skel_xyz = skel_xyzr[:, :, :-1].cpu().squeeze()
skel_radius = skel_xyzr[:, :, -1].cpu().squeeze()

mat_edges = model.mat_edges.cpu().squeeze()
mat_faces = model.mat_faces.cpu().squeeze()
write_ma('test.ma', skel_xyz, skel_radius, mat_edges, mat_faces)
