import torch.nn as nn
import torch
from mat.mat_handling import get_surface_points

import lightning as pl

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance

from itertools import chain
from model.graph_conv import adjacency_matrix, Features2Features, Feature2VertexLayer
from model.feature_sampling import LearntNeighbourhoodSampling
from utils.file_handle import read_ma
import torch.optim as optim
import numpy as np


class LitVoxel2Mesh(pl.LightningModule):
    def __init__(self, v2m_model, config):
        super().__init__()
        self.save_hyperparameters(ignore=['v2m_model'])
        self.model = v2m_model
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

        target_segmentation = batch['segmentation']['data'].squeeze(0).long()
        predicted_segmentation = pred[0][-1][2]

        cross_entropy_loss = nn.CrossEntropyLoss()
        ce_loss = cross_entropy_loss(predicted_segmentation, target_segmentation)

        chamfer_loss = torch.tensor(0).float().to(pred[0][-1][2].device)

        for c in range(self.config.num_classes - 1):
            target_surface = batch['surface_vtx'][c]
            for k, (vertices, _, _, _) in enumerate(pred[c][1:]):
                vertices = vertices * self.model.mat_features_scalar.to(vertices.device)
                pred_points = get_surface_points(vertices[:, :, :-1].squeeze(), vertices[:, :, -1].squeeze(),
                                                 self.model.mat_edges.squeeze(), self.model.mat_faces.squeeze(),
                                                 self.model.mat_lines.squeeze(), device=vertices.device)

                chamfer_loss += chamfer_distance(pred_points, target_surface)[0]

        loss = 1 * chamfer_loss + 1 * ce_loss
        self.training_targets.append(target_surface)
        self.training_outputs.append(pred)
        self.training_losses.append([loss.item(), chamfer_loss.item(), ce_loss.item()])
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['volume']['data']
        pred = self.model(x)
        target_segmentation = batch['segmentation']['data'].squeeze(0).long()
        predicted_segmentation = pred[0][-1][2]

        cross_entropy_loss = nn.CrossEntropyLoss()
        ce_loss = cross_entropy_loss(predicted_segmentation, target_segmentation)

        chamfer_loss = torch.tensor(0).float().to(pred[0][-1][2].device)

        for c in range(self.config.num_classes - 1):
            target_surface = batch['surface_vtx'][c]
            for k, (vertices, _, _, _) in enumerate(pred[c][1:]):
                vertices = vertices * self.model.mat_features_scalar.to(vertices.device)
                pred_points = get_surface_points(vertices[:, :, :-1].squeeze(), vertices[:, :, -1].squeeze(),
                                                 self.model.mat_edges.squeeze(), self.model.mat_faces.squeeze(),
                                                 self.model.mat_lines.squeeze(), device=vertices.device)

                chamfer_loss += chamfer_distance(pred_points, target_surface)[0]

        loss = 1 * chamfer_loss + 1 * ce_loss
        self.validation_targets.append(target_surface)
        self.validation_outputs.append(pred)
        self.validation_losses.append([loss.item(), chamfer_loss.item(), ce_loss.item()])

    def on_train_epoch_end(self):
        loss_mean, chamfer_loss_mean, ce_loss_mean = np.mean(self.training_losses, axis=0)
        tensorboard = self.logger.experiment
        tensorboard.add_scalar('Loss/train', loss_mean, self.current_epoch)
        tensorboard.add_scalar('Chamfer Loss/train', chamfer_loss_mean, self.current_epoch)
        tensorboard.add_scalar('Cross Entropy Loss/train', ce_loss_mean, self.current_epoch)
        m = 0
        for lis in self.training_outputs:
            tensorboard.add_mesh('Mesh {:n}'.format(m), vertices=lis[0][-1][0][:,:,:-1], faces=self.model.mat_faces, global_step=self.current_epoch)

            m += 1
        self.training_outputs.clear()
        self.training_losses.clear()
        self.training_targets.clear()

    def on_validation_epoch_end(self):
        loss_mean, chamfer_loss_mean, ce_loss_mean = np.mean(self.validation_losses, axis=0)
        tensorboard = self.logger.experiment
        tensorboard.add_scalar('Loss/validation', loss_mean, self.current_epoch)
        tensorboard.add_scalar('Chamfer Loss/validation', chamfer_loss_mean, self.current_epoch)
        tensorboard.add_scalar('Cross Entropy Loss/validation', ce_loss_mean, self.current_epoch)
        self.log("val_loss", loss_mean)


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
    def __init__(self, num_channels_in, num_channels_out, ndims, batch_norm=False):
        super(UNetLayer, self).__init__()
        conv_op = nn.Conv2d if ndims == 2 else nn.Conv3d
        batch_nrom_op = nn.BatchNorm2d if ndims == 2 else nn.BatchNorm3d

        conv1 = conv_op(num_channels_in,  num_channels_out, kernel_size=3, padding=1)
        conv2 = conv_op(num_channels_out, num_channels_out, kernel_size=3, padding=1)

        bn1 = batch_nrom_op(num_channels_out)
        bn2 = batch_nrom_op(num_channels_out)
        self.unet_layer = nn.Sequential(conv1, bn1, nn.ReLU(), conv2, bn2, nn.ReLU())

    def forward(self, x):
        return self.unet_layer(x)


class Voxel2Mesh(nn.Module):
    """ Voxel2Mesh  """
    def __init__(self, config):
        super(Voxel2Mesh, self).__init__()
        self.config = config
        self.max_pool = nn.MaxPool3d(2) if config.ndims == 3 else nn.MaxPool2d(2)
        ConvLayer = nn.Conv3d if config.ndims == 3 else nn.Conv2d
        ConvTransposeLayer = nn.ConvTranspose3d if config.ndims == 3 else nn.ConvTranspose2d

        '''  Down layers '''
        down_layers = [UNetLayer(config.num_input_channels, config.first_layer_channels, config.ndims)]
        for i in range(1, config.steps + 1):
            graph_conv_layer = UNetLayer(config.first_layer_channels * 2 ** (i - 1), config.first_layer_channels * 2 ** i, config.ndims)
            down_layers.append(graph_conv_layer)
        self.down_layers = down_layers
        self.encoder = nn.Sequential(*down_layers)

        ''' Up layers '''
        self.skip_count = []
        self.latent_features_coount = []
        for i in range(config.steps+1):
            self.skip_count += [config.first_layer_channels * 2 ** (config.steps-i)]
            self.latent_features_coount += [32]

        dim = 4

        up_std_conv_layers = []
        up_f2f_layers = []
        up_f2v_layers = []
        for i in range(config.steps+1):
            graph_unet_layers = []
            feature2vertex_layers = []
            skip = LearntNeighbourhoodSampling(config, self.skip_count[i], i)
            if i == 0:
                grid_upconv_layer = None
                grid_unet_layer = None
                for k in range(config.num_classes-1):
                    graph_unet_layers += [Features2Features(self.skip_count[i] + dim, self.latent_features_coount[i], hidden_layer_count=config.graph_conv_layer_count)] # , graph_conv=GraphConv

            else:
                grid_upconv_layer = ConvTransposeLayer(in_channels=config.first_layer_channels * 2**(config.steps - i+1), out_channels=config.first_layer_channels * 2**(config.steps-i), kernel_size=2, stride=2)
                grid_unet_layer = UNetLayer(config.first_layer_channels * 2**(config.steps - i + 1), config.first_layer_channels * 2**(config.steps-i), config.ndims, config.batch_norm)
                for k in range(config.num_classes-1):
                    graph_unet_layers += [Features2Features(self.skip_count[i] + self.latent_features_coount[i-1] + dim, self.latent_features_coount[i], hidden_layer_count=config.graph_conv_layer_count)] #, graph_conv=GraphConv if i < config.steps else GraphConvNoNeighbours

            for k in range(config.num_classes-1):
                feature2vertex_layers += [Feature2VertexLayer(self.latent_features_coount[i], 3)]


            up_std_conv_layers.append((skip, grid_upconv_layer, grid_unet_layer))
            up_f2f_layers.append(graph_unet_layers)
            up_f2v_layers.append(feature2vertex_layers)



        self.up_std_conv_layers = up_std_conv_layers
        self.up_f2f_layers = up_f2f_layers
        self.up_f2v_layers = up_f2v_layers

        self.decoder_std_conv = nn.Sequential(*chain(*up_std_conv_layers))
        self.decoder_f2f = nn.Sequential(*chain(*up_f2f_layers))
        self.decoder_f2v = nn.Sequential(*chain(*up_f2v_layers))

        ''' Final layer (for voxel decoder)'''
        self.final_layer = ConvLayer(in_channels=config.first_layer_channels, out_channels=config.num_classes, kernel_size=1)

        mat_path = config.skl_path
        vertices, radii, mat_edges, mat_faces, mat_lines = read_ma(mat_path)
        mat_features = torch.cat([vertices, radii.unsqueeze(1)], dim=1)
        mat_features = mat_features.float()

        self.mat_features = (mat_features/torch.sqrt(torch.sum(mat_features**2, dim=0))).unsqueeze(0)
        self.mat_features_scalar = torch.tensor([mat_features[:,0].max()*10, mat_features[:,1].max()*10, mat_features[:,2].max()*10, mat_features[:,3].max()])
        # self.mat_features = mat_features.unsqueeze(0) #se non vuoi normalizzare
        self.mat_edges = mat_edges.long().unsqueeze(0)
        self.mat_faces = mat_faces.long().unsqueeze(0)
        self.mat_lines = mat_lines.long().unsqueeze(0)
        self.A, self.D = adjacency_matrix(self.mat_features, self.mat_edges)

    def forward(self, x):
        vertices_features = self.mat_features.clone().to(x.device)
        vertices = vertices_features.clone()
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
            pred[k] = [[vertices.clone(), None, None, vertices_features.clone()]]

 
        for i, ((skip_connection, grid_upconv_layer, grid_unet_layer), up_f2f_layers, up_f2v_layers, down_output, skip_amount) in enumerate(zip(self.up_std_conv_layers, self.up_f2f_layers, self.up_f2v_layers, down_outputs[::-1], self.skip_count)):
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
                graph_unet_layer = up_f2f_layers[k]
                feature2vertex = up_f2v_layers[k]

                skipped_features = skip_connection(x[:, :skip_amount], vertices)      
                      
                latent_features = torch.cat([latent_features, skipped_features, vertices], dim=2) if latent_features is not None else torch.cat([skipped_features, vertices], dim=2)
 
                latent_features = graph_unet_layer(latent_features, self.A.to(latent_features.device), self.D.to(latent_features.device))
                deltaV = feature2vertex(latent_features, self.A.to(latent_features.device), self.D.to(latent_features.device))
                vertices = vertices + deltaV


                voxel_pred = self.final_layer(x) if i == len(self.up_std_conv_layers)-1 else None

                pred[k] += [[vertices, latent_features, voxel_pred, vertices_features]]
        return pred
