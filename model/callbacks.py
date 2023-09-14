from lightning.pytorch.callbacks import Callback
import pyvista as pv
import numpy as np

class GifCallback(Callback):
    def __init__(self):
        self.plotters = []

    def on_train_start(self, trainer, pl_module):
        print("Training is starting (>‘o’)>")
        for mesh_n in range(len(trainer.train_dataloader)):
            self.plotters.append(pv.Plotter(notebook=False, off_screen=True))
            self.plotters[mesh_n].open_gif("mesh_{:n}.gif".format(mesh_n))
            faces_padded = []
            for face in pl_module.mat_faces.squeeze().numpy():
                num_points = len(face)
                faces_padded.extend([num_points, *face])
            faces_padded = np.array(faces_padded, dtype=np.int32)

            lines_padded = []
            for line in pl_module.mat_lines.squeeze().numpy():
                num_points = len(line)
                lines_padded.extend([num_points, *line])
            lines_padded = np.array(lines_padded, dtype=np.int32)
            self.skl_mesh = pv.PolyData(pl_module.mat_features.squeeze().numpy()[:, :-1], faces=faces_padded,
                                        lines=lines_padded)

    def on_train_epoch_end(self, trainer, pl_module):
        mesh_n = 0
        for lis in pl_module.training_outputs:
            for c in range(pl_module.config.num_classes - 1):
                target_surface = pl_module.training_targets[mesh_n][1][c].detach().cpu().numpy()

                skl_mesh = self.skl_mesh.copy()
                vtx = lis[1][-1][0].detach().squeeze().cpu().numpy()
                skl_mesh.points = vtx[:, :-1]
                self.plotters[mesh_n].add_text("Target", font_size=30, color='#FF0000', position='upper_right')
                self.plotters[mesh_n].add_points(target_surface, color='#FF0000', opacity=0.20)

                self.plotters[mesh_n].add_text("X", font_size=30, color='#00FF00')
                self.plotters[mesh_n].add_mesh(skl_mesh, color='#00FF00')
                for point, rad in zip(skl_mesh.points, vtx[:, -1]):
                    self.plotters[mesh_n].add_mesh(pv.Sphere(center=point, radius=rad), opacity=0.20)
                # plotter.add_points(pred_points.detach().cpu().numpy())
                self.plotters[mesh_n].write_frame()
                self.plotters[mesh_n].clear()
            mesh_n += 1

    def on_train_end(self, trainer, pl_module):
        print("Training is ending ≧◡≦")
        for mesh_n in range(len(trainer.train_dataloader)):
            self.plotters[mesh_n].close()
        print("Training is finished!!! ᕙ(^▿^-ᕙ)")
class PyVistaGifCallback(Callback):
    def __init__(self):
        self.skl_mesh = None
        self.plotters = []

    def on_train_start(self, trainer, pl_module):
        print("Training is starting (>‘o’)>")
        for mesh_n in range(len(trainer.train_dataloader)):
            self.plotters.append(pv.Plotter(notebook=False, off_screen=True))
            self.plotters[mesh_n].open_gif("mesh_{:n}.gif".format(mesh_n))
        faces_padded = []
        for face in pl_module.model.mat_faces.squeeze().numpy():
            num_points = len(face)
            faces_padded.extend([num_points, *face])
        faces_padded = np.array(faces_padded, dtype=np.int32)

        lines_padded = []
        for line in pl_module.model.mat_lines.squeeze().numpy():
            num_points = len(line)
            lines_padded.extend([num_points, *line])
        lines_padded = np.array(lines_padded, dtype=np.int32)
        self.skl_mesh = pv.PolyData(pl_module.model.mat_features.squeeze().numpy()[:, :-1], faces=faces_padded, lines=lines_padded)

    def on_train_epoch_end(self, trainer, pl_module):
        mesh_n = 0
        for lis in pl_module.training_outputs:
            for c in range(pl_module.config.num_classes - 1):
                target_surface = pl_module.training_mesh_target[mesh_n][c].detach().cpu().numpy()

                skl_mesh = self.skl_mesh.copy()
                vtx = lis[0][-1][0][:, :, :].detach().squeeze().cpu().numpy() * pl_module.model.mat_features_scalar.cpu().numpy()
                skl_mesh.points = vtx[:, :-1]
                self.plotters[mesh_n].add_text("Target", font_size=30, color='#FF0000', position='upper_right')
                self.plotters[mesh_n].add_points(target_surface, color='#FF0000', opacity=0.20)

                self.plotters[mesh_n].add_text("X", font_size=30, color='#00FF00')
                self.plotters[mesh_n].add_mesh(skl_mesh, color='#00FF00')
                for point, rad in zip(skl_mesh.points, vtx[:, -1]):
                    self.plotters[mesh_n].add_mesh(pv.Sphere(center=point, radius=rad), opacity=0.20)
                # plotter.add_points(pred_points.detach().cpu().numpy())
                self.plotters[mesh_n].write_frame()
                self.plotters[mesh_n].clear()
            mesh_n += 1

    def on_train_end(self, trainer, pl_module):
        print("Training is ending ≧◡≦")
        for mesh_n in range(len(trainer.train_dataloader)):
            self.plotters[mesh_n].close()
        print("Training is finished!!! ᕙ(^▿^-ᕙ)")
