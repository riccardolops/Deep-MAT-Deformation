import torch.nn as nn
import torch
from mat.mat_handling import MATMeshSurface
import lightning as pl
import model.DistanceFunction as DF
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
import torch.nn.functional as F
from itertools import chain
from model.graph_conv import adjacency_matrix, Features2Features, Feature2VertexLayer
from model.feature_sampling import LearntNeighbourhoodSampling
from utils.file_handle import read_ma
import torch.optim as optim
import numpy as np
import wandb


class LitVoxel2Mesh(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(ignore=['v2m_model'])
        self.model = Voxel2Mesh(config)
        self.config = config
        self.training_outputs = []
        self.training_losses = []
        self.training_targets = []
        self.validation_targets = []
        self.validation_outputs = []
        self.validation_losses = []

    def training_step(self, batch):
        x = batch['volume']['data']
        pred = self.model(x)
        ce_loss, loss_dice, loss_sample, loss_point2sphere, loss_radius, target_segmentation = self.model.compute_loss(pred, batch)
        loss = loss_sample + 0.8 * loss_point2sphere + 0.1 * loss_radius + 50 * ce_loss + 50 * loss_dice
        self.training_targets.append(target_segmentation)
        self.training_outputs.append(pred)
        self.training_losses.append([loss.item(), loss_sample.item(), loss_point2sphere.item(), loss_radius.item(), ce_loss.item(), loss_dice.item()])
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['volume']['data']
        pred = self.model(x)
        ce_loss, loss_dice, loss_sample, loss_point2sphere, loss_radius, target_segmentation = self.model.compute_loss(pred, batch)
        loss = loss_sample + 0.8 * loss_point2sphere + 0.1 * loss_radius + 50 * ce_loss + 50 * loss_dice
        self.validation_targets.append(target_segmentation)
        self.validation_outputs.append(pred)
        self.validation_losses.append([loss.item(), loss_sample.item(), loss_point2sphere.item(), loss_radius.item(), ce_loss.item(), loss_dice.item()])

    def on_train_epoch_end(self):
        loss_mean, loss_sample_mean, loss_point2sphere_mean, loss_radius_mean, ce_loss_mean, dice_loss_mean = np.mean(self.training_losses, axis=0)
        values = {"loss": loss_mean, "sample": loss_sample_mean, "Point2Sphere": loss_point2sphere_mean, "Radius": loss_radius_mean, "Cross Entropy": ce_loss_mean, "Dice Loss": dice_loss_mean}
        self.log_dict(values)
        for patient_pred, patient_target in zip(self.training_outputs, self.training_targets):
            pred_tensor = patient_pred[0][-1][2]
            class_predictions = torch.argmax(pred_tensor, dim=1)
            segmentation_mask = (class_predictions == 1).float()
            wandb.log({
                "Predicted": [wandb.Image(np.uint8(segmentation_mask[0][50].cpu().numpy() * 255))],
                "Target": [wandb.Image(np.uint8(patient_target[0][50].cpu().numpy() * 255))]
            })


        self.training_outputs.clear()
        self.training_losses.clear()
        self.training_targets.clear()

    def on_validation_epoch_end(self):
        loss_mean, loss_sample_mean, loss_point2sphere_mean, loss_radius_mean, ce_loss_mean, dice_loss_mean = np.mean(self.validation_losses, axis=0)
        values = {"val_loss": loss_mean, "val_sample": loss_sample_mean, "val_Point2Sphere": loss_point2sphere_mean, "val_Radius": loss_radius_mean, "val_Cross Entropy": ce_loss_mean, "val_Dice Loss": dice_loss_mean}
        self.log_dict(values)
        self.validation_targets.clear()
        self.validation_outputs.clear()
        self.validation_losses.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config.learning_rate)
        return optimizer


def crop_slices(shape1, shape2):
    slices = [slice((sh1 - sh2) // 2, (sh1 - sh2) // 2 + sh2) for sh1, sh2 in zip(shape1, shape2)]
    return slices


def crop_and_merge(tensor1, tensor2):
    slices = crop_slices(tensor1.size(), tensor2.size())
    slices[0] = slice(None)
    slices[1] = slice(None)
    slices = tuple(slices)

    return torch.cat((tensor1[slices], tensor2), 1)


class UNetLayer(nn.Module):
    """ U-Net Layer """
    def __init__(self, num_channels_in, num_channels_out):
        super(UNetLayer, self).__init__()

        conv1 = nn.Conv3d(num_channels_in,  num_channels_out, kernel_size=3, padding=1)
        conv2 = nn.Conv3d(num_channels_out, num_channels_out, kernel_size=3, padding=1)

        bn1 = nn.BatchNorm3d(num_channels_out)
        bn2 = nn.BatchNorm3d(num_channels_out)

        self.unet_layer = nn.Sequential(conv1, bn1, nn.ReLU(), conv2, bn2, nn.ReLU())

    def forward(self, x):
        return self.unet_layer(x)


class Voxel2Mesh(nn.Module):
    """ Voxel2Mesh  """
    def __init__(self, config):
        super(Voxel2Mesh, self).__init__()
        self.config = config
        self.max_pool = nn.MaxPool3d(2)

        '''  Down layers '''
        self.down_layers = nn.ModuleList()
        self.down_layers.append(UNetLayer(config.num_input_channels, config.first_layer_channels))
        for i in range(1, config.steps + 1):
            graph_conv_layer = UNetLayer(config.first_layer_channels * 2 ** (i - 1), config.first_layer_channels * 2 ** i)
            self.down_layers.append(graph_conv_layer)

        ''' Up layers '''
        self.skip_count = []
        self.latent_features_count = []
        for i in range(config.steps+1):
            self.skip_count += [config.first_layer_channels * 2 ** (config.steps-i)]
            self.latent_features_count += [32]

        dim = 4

        self.up_f2f_layers = nn.ModuleList()
        self.up_f2v_layers = nn.ModuleList()
        self.skip_connection = nn.ModuleList()
        self.grid_upconv_layer = nn.ModuleList()
        self.grid_unet_layer = nn.ModuleList()
        for i in range(config.steps+1):
            feature2feature_layers = nn.ModuleList()
            feature2vertex_layers = nn.ModuleList()
            self.skip_connection.append(LearntNeighbourhoodSampling(config, self.skip_count[i], i))
            if i == 0:
                self.grid_upconv_layer.append(None)
                self.grid_unet_layer.append(None)
                for k in range(config.num_classes-1):
                    feature2feature_layers.append(Features2Features(self.skip_count[i] + dim, self.latent_features_count[i], hidden_layer_count=config.graph_conv_layer_count))  # , graph_conv=GraphConv

            else:
                self.grid_upconv_layer.append(nn.ConvTranspose3d(in_channels=config.first_layer_channels * 2**(config.steps - i+1), out_channels=config.first_layer_channels * 2**(config.steps-i), kernel_size=2, stride=2))
                self.grid_unet_layer.append(UNetLayer(config.first_layer_channels * 2**(config.steps - i + 1), config.first_layer_channels * 2**(config.steps-i)))
                for k in range(config.num_classes-1):
                    feature2feature_layers.append(Features2Features(self.skip_count[i] + self.latent_features_count[i - 1] + dim, self.latent_features_count[i], hidden_layer_count=config.graph_conv_layer_count))  #, graph_conv=GraphConv if i < config.steps else GraphConvNoNeighbours

            for k in range(config.num_classes-1):
                feature2vertex_layers.append(Feature2VertexLayer(self.latent_features_count[i], 3))

            self.up_f2f_layers.append(feature2feature_layers)
            self.up_f2v_layers.append(feature2vertex_layers)


        ''' Final layer (for voxel decoder)'''
        self.final_layer = nn.Conv3d(in_channels=config.first_layer_channels, out_channels=config.num_classes, kernel_size=1)

        mat_path = config.skl_path
        vertices, radii, mat_edges, mat_faces, mat_lines = read_ma(mat_path)
        mat_features = torch.cat([vertices, radii.unsqueeze(1)], dim=1)
        mat_features = mat_features.float()

        #self.mat_features = (mat_features/torch.sqrt(torch.sum(mat_features**2, dim=0))).unsqueeze(0)
        #self.mat_features_scalar = nn.Parameter(torch.sqrt(torch.sum(mat_features**2, dim=0)), requires_grad=False)

        self.mat_features = (mat_features / mat_features.max()).unsqueeze(0)
        self.mat_features_scalar = nn.Parameter(mat_features.max(), requires_grad=False)
        self.centroid = mat_features[:, :3].mean(dim=0)

        # self.mat_features = mat_features.unsqueeze(0) #se non vuoi normalizzare
        self.mat_edges = mat_edges.long().unsqueeze(0)
        self.mat_faces = mat_faces.long().unsqueeze(0)
        self.mat_lines = mat_lines.long().unsqueeze(0)
        A, D = adjacency_matrix(self.mat_features, self.mat_edges)
        self.A = nn.Parameter(A, requires_grad=False)
        self.D = nn.Parameter(D, requires_grad=False)
        self.average_segm_pos = nn.Parameter(torch.tensor([148., 174., 66.]), requires_grad=False)
        self.get_surface_points = MATMeshSurface(self.mat_edges.squeeze(), self.mat_faces.squeeze(), self.mat_lines.squeeze())
        self.ce_weight = nn.Parameter(torch.tensor([0.5, 2.0]), requires_grad=False)

    def forward(self, x):
        vertices_features = self.mat_features.clone().to(x.device)
        batch_size = self.config.batch_size
 
        # first layer
        x = self.down_layers[0](x)
        down_outputs = [x]

        # down layers
        for unet_layer in self.down_layers[1:]:
            x = self.max_pool(x)
            x = unet_layer(x) 
            down_outputs.append(x)

  
        pred = [None] * self.config.num_classes
        for k in range(self.config.num_classes-1):
            pred[k] = [[vertices_features.clone(), None, None, vertices_features.clone()]]

 
        for i, (skip_connection, grid_upconv_layer, grid_unet_layer, up_f2f_layers, up_f2v_layers, down_output, skip_amount) in enumerate(zip(self.skip_connection, self.grid_upconv_layer, self.grid_unet_layer, self.up_f2f_layers, self.up_f2v_layers, down_outputs[::-1], self.skip_count)):
            if grid_upconv_layer is not None and i > 0:
                x = grid_upconv_layer(x)
                x = crop_and_merge(down_output, x)
                x = grid_unet_layer(x)
            elif grid_upconv_layer is None:
                x = down_output
          

            for k in range(self.config.num_classes-1):

                # load mesh information from previous iteration for class k
                vertices = pred[k][i][0]
                latent_features = pred[k][i][1]
                vertices_features = pred[k][i][3]
                feature2feature_layer = up_f2f_layers[k]
                feature2vertex = up_f2v_layers[k]

                skipped_features = skip_connection(x[:, :skip_amount], vertices)      
                      
                latent_features = torch.cat([latent_features, skipped_features, vertices], dim=2) if latent_features is not None else torch.cat([skipped_features, vertices], dim=2)
 
                latent_features = feature2feature_layer(latent_features, self.A, self.D)
                deltaV = feature2vertex(latent_features, self.A, self.D)
                vertices = vertices + deltaV

                voxel_pred = self.final_layer(x) if i == len(self.skip_connection)-1 else None

                pred[k] += [[vertices, latent_features, voxel_pred, vertices_features]]
        return pred

    def compute_loss(self, pred, batch):
        target_segmentation = batch['segmentation']['data'].squeeze(0).long()
        predicted_segmentation = pred[0][-1][2]

        cross_entropy_loss = nn.CrossEntropyLoss(weight=self.ce_weight)
        ce_loss = cross_entropy_loss(predicted_segmentation, target_segmentation)
        outputs_soft = F.softmax(predicted_segmentation, dim=1)
        def dice_loss(score, target):
            target = target.float()
            smooth = 1e-5
            intersect = torch.sum(score * target)
            y_sum = torch.sum(target * target)
            z_sum = torch.sum(score * score)
            loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
            loss = 1 - loss
            return loss
        loss_dice = dice_loss(outputs_soft[:, 1, :, :, :], target_segmentation == 1)
        for c in range(self.config.num_classes - 1):
            shape_xyz = batch['surface_vtx'][c]
            shape_xyz = shape_xyz - shape_xyz.mean(dim=1) + self.centroid.to('cuda')
            (vertices, _, _, _) = pred[c][1:][0]
            skel_xyzr = vertices * self.mat_features_scalar
            #pred_points = self.get_surface_points(skel_xyzr.squeeze())
            #chamfer_loss = chamfer_distance(skel_xyzr[:,:,:-1], shape_xyz)[0]
            skel_xyz = skel_xyzr[:,:,:-1]
            skel_radius = skel_xyzr[:,:,-1]
            bn = skel_xyz.size()[0]
            shape_pnum = float(shape_xyz.size()[1])
            skel_pnum = float(skel_xyz.size()[1])
            e = 0.57735027
            sample_directions = torch.tensor(
                [[e, e, e], [e, e, -e], [e, -e, e], [e, -e, -e], [-e, e, e], [-e, e, -e], [-e, -e, e], [-e, -e, -e]])
            sample_directions = torch.unsqueeze(sample_directions, 0)
            sample_directions = sample_directions.repeat(bn, int(skel_pnum), 1).cuda()
            sample_centers = torch.repeat_interleave(skel_xyz, 8, dim=1)
            sample_radius = torch.repeat_interleave(skel_radius, 8, dim=1)
            sample_xyz = sample_centers + sample_radius.view(bn, int(8*skel_pnum), 1) * sample_directions

            cd_sample1 = DF.closest_distance_with_batch(sample_xyz, shape_xyz) / (skel_pnum * 8)
            cd_sample2 = DF.closest_distance_with_batch(shape_xyz, sample_xyz) / (shape_pnum)
            loss_sample = cd_sample1 + cd_sample2

            cd_point2pshere1 = DF.point2sphere_distance_with_batch(shape_xyz, skel_xyzr) / shape_pnum
            cd_point2sphere2 = DF.sphere2point_distance_with_batch(skel_xyzr, shape_xyz) / skel_pnum
            loss_point2sphere = cd_point2pshere1 + cd_point2sphere2
            loss_radius = - torch.sum(skel_radius) / skel_pnum

            for k, (vertices, _, _, _) in enumerate(pred[c][1:][1:]):
                skel_xyzr = vertices * self.mat_features_scalar
                #pred_points = self.get_surface_points(skel_xyzr.squeeze())
                #chamfer_loss += chamfer_distance(skel_xyzr[:,:,:-1], shape_xyz)[0]
                skel_xyz = skel_xyzr[:, :, :-1]
                skel_radius = skel_xyzr[:, :, -1]
                bn = skel_xyz.size()[0]
                shape_pnum = float(shape_xyz.size()[1])
                skel_pnum = float(skel_xyz.size()[1])
                e = 0.57735027
                sample_directions = torch.tensor(
                    [[e, e, e], [e, e, -e], [e, -e, e], [e, -e, -e], [-e, e, e], [-e, e, -e], [-e, -e, e],
                     [-e, -e, -e]])
                sample_directions = torch.unsqueeze(sample_directions, 0)
                sample_directions = sample_directions.repeat(bn, int(skel_pnum), 1).cuda()
                sample_centers = torch.repeat_interleave(skel_xyz, 8, dim=1)
                sample_radius = torch.repeat_interleave(skel_radius, 8, dim=1)
                sample_xyz = sample_centers + sample_radius.view(bn, int(8*skel_pnum), 1) * sample_directions

                cd_sample1 = DF.closest_distance_with_batch(sample_xyz, shape_xyz) / (skel_pnum * 8)
                cd_sample2 = DF.closest_distance_with_batch(shape_xyz, sample_xyz) / (shape_pnum)
                loss_sample += cd_sample1 + cd_sample2

                cd_point2pshere1 = DF.point2sphere_distance_with_batch(shape_xyz, skel_xyzr) / shape_pnum
                cd_point2sphere2 = DF.sphere2point_distance_with_batch(skel_xyzr, shape_xyz) / skel_pnum
                loss_point2sphere += cd_point2pshere1 + cd_point2sphere2
                loss_radius += - torch.sum(skel_radius) / skel_pnum

        return ce_loss, loss_dice, loss_sample, loss_point2sphere, loss_radius, target_segmentation