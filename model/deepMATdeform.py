import torch.nn as nn
import torch
from model.graph_conv import Features2Features, Feature2VertexLayer
from model.feature_sampling import LearntNeighbourhoodSampling
import torch.nn.functional as F
import model.DistanceFunction as DF
import numpy as np
import torch.optim as optim
from utils.file_handle import read_ma
from model.graph_conv import adjacency_matrix
import wandb
import lightning.pytorch as pl



class LitVoxel2MAT(pl.LightningModule):
    """ LitVoxel2MAT Lightning Module """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cnn_network = UNetModule(config)
        self.ce_weight = nn.Parameter(torch.tensor([0.5, 2.0]), requires_grad=False)

        self.training_losses = []
        self.training_targets = []
        self.training_outputs = []

        self.validation_losses = []
        self.validation_targets = []
        self.validation_outputs = []

        vertices, radii, mat_edges, mat_faces, mat_lines = read_ma(config.skl_path)
        mat_features = torch.cat([vertices - vertices.mean(0), radii.unsqueeze(1)], dim=1)
        mat_features = mat_features.float()

        self.mat_features = (mat_features - mat_features.mean(dim=0)).unsqueeze(0)
        #self.mat_features_scalar = nn.Parameter(mat_features.max(), requires_grad=False)

        self.mat_edges = mat_edges.long().unsqueeze(0)
        self.mat_faces = mat_faces.long().unsqueeze(0)
        self.mat_lines = mat_lines.long().unsqueeze(0)
        A, D = adjacency_matrix(self.mat_features, self.mat_edges)
        A = nn.Parameter(A, requires_grad=False)
        D = nn.Parameter(D, requires_grad=False)
        self.graph_network = MATDecoder(config, A, D)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        volume_data = batch['image']
        voxel_pred, MAT_deformed = self(volume_data)
        mask = (torch.argmax(F.softmax(voxel_pred, dim=1), dim=1)).squeeze(0)
        return mask, MAT_deformed, batch['surface_vtx'][0]

    def forward(self, volume_data):
        """ Input : volume_data, Output : voxel_pred, MAT_deformed """
        voxel_pred, decoder_features = self.cnn_network(volume_data)
        mask = (torch.argmax(F.softmax(voxel_pred, dim=1), dim=1)).squeeze(0)
        centroid = first_moment(mask)
        origin_vtx = self.mat_features.clone().to(volume_data.device)
        centroid_shaped = centroid.view(1, 1, 3)
        vtx = torch.cat((origin_vtx[..., :3] + centroid_shaped, origin_vtx[..., 3:]), dim=-1)

        MAT_deformed = self.graph_network(vtx, decoder_features)
        return voxel_pred, MAT_deformed

    def loss(self, voxel_pred, MAT_deformed, batch):
        target_segmentation = batch['label'].squeeze(0).long()
        shape_xyz = batch['surface_vtx'][0]
        cross_entropy_loss = nn.CrossEntropyLoss(weight=self.ce_weight)
        ce_loss = cross_entropy_loss(voxel_pred, target_segmentation)
        outputs_soft = F.softmax(voxel_pred, dim=1)

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


        vertices = MAT_deformed[0]
        skel_xyzr = vertices # * self.mat_features_scalar
        skel_xyz = skel_xyzr[:, :, :-1]
        skel_radius = skel_xyzr[:, :, -1]
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
        sample_xyz = sample_centers + sample_radius.view(bn, int(8 * skel_pnum), 1) * sample_directions

        cd_sample1 = DF.closest_distance_with_batch(sample_xyz, shape_xyz) / (skel_pnum * 8)
        cd_sample2 = DF.closest_distance_with_batch(shape_xyz, sample_xyz) / (shape_pnum)
        loss_sample = cd_sample1 + cd_sample2

        cd_point2pshere1 = DF.point2sphere_distance_with_batch(shape_xyz, skel_xyzr) / shape_pnum
        cd_point2sphere2 = DF.sphere2point_distance_with_batch(skel_xyzr, shape_xyz) / skel_pnum
        loss_point2sphere = cd_point2pshere1 + cd_point2sphere2
        loss_radius = - torch.sum(skel_radius) / skel_pnum

        for vertices in MAT_deformed[1:]:
            skel_xyzr = vertices# * self.mat_features_scalar
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
            sample_xyz = sample_centers + sample_radius.view(bn, int(8 * skel_pnum), 1) * sample_directions

            cd_sample1 = DF.closest_distance_with_batch(sample_xyz, shape_xyz) / (skel_pnum * 8)
            cd_sample2 = DF.closest_distance_with_batch(shape_xyz, sample_xyz) / (shape_pnum)
            loss_sample += cd_sample1 + cd_sample2

            cd_point2pshere1 = DF.point2sphere_distance_with_batch(shape_xyz, skel_xyzr) / shape_pnum
            cd_point2sphere2 = DF.sphere2point_distance_with_batch(skel_xyzr, shape_xyz) / skel_pnum
            loss_point2sphere += cd_point2pshere1 + cd_point2sphere2
            loss_radius += - torch.sum(skel_radius) / skel_pnum

        return ce_loss, loss_dice, loss_sample, loss_point2sphere, loss_radius

    def training_step(self, batch):
        volume_data = batch['image']
        voxel_pred, MAT_deformed = self(volume_data)
        self.training_outputs.append([voxel_pred, MAT_deformed])
        ce_loss, loss_dice, loss_sample, loss_point2sphere, loss_radius = self.loss(voxel_pred, MAT_deformed, batch)
        loss = self.config.lambda_sample * loss_sample + self.config.lambda_p2s * loss_point2sphere + self.config.lambda_radius * loss_radius + self.config.lambda_ce * ce_loss + self.config.lambda_dice * loss_dice
        self.training_losses.append([
            loss.item(),
            loss_sample.item(),
            loss_point2sphere.item(),
            loss_radius.item(),
            ce_loss.item(),
            loss_dice.item()
        ])

        target_segmentation = batch['label'].squeeze(0).long()
        shape_xyz = batch['surface_vtx'][0]
        self.training_targets.append([target_segmentation, shape_xyz])
        
        return loss

    def on_train_epoch_end(self):
        loss_mean, loss_sample_mean, loss_point2sphere_mean, loss_radius_mean, ce_loss_mean, dice_loss_mean = np.mean(
            self.training_losses, axis=0)
        values = {"loss": loss_mean, "sample": loss_sample_mean, "Point2Sphere": loss_point2sphere_mean,
                  "Radius": loss_radius_mean, "Cross Entropy": ce_loss_mean, "Dice Loss": dice_loss_mean}
        self.log_dict(values)

        for patient_pred, patient_target in zip(self.training_outputs, self.training_targets):
            pred_tensor = patient_pred[0]
            class_predictions = torch.argmax(F.softmax(pred_tensor, dim=1), dim=1)
            segmentation_mask = (class_predictions == 1).float()
            wandb.log({
                "Predicted": [wandb.Image(np.uint8(segmentation_mask[0][50].cpu().numpy() * 255))],
                "Target": [wandb.Image(np.uint8(patient_target[0][0][50].cpu().numpy() * 255))]
            })
            wandb.log({"3D Target": wandb.Object3D({
                "type": "lidar/beta",
                "points": patient_target[1].squeeze(0).cpu().numpy(),
            })})
            wandb.log({"3D Prediction": wandb.Object3D({
                "type": "lidar/beta",
                "points": (patient_pred[1][-1].squeeze())[:, :3].detach().cpu().numpy()
            })})

        self.training_outputs.clear()
        self.training_losses.clear()
        self.training_targets.clear()

    def validation_step(self, batch, batch_idx):
        volume_data = batch['image']
        voxel_pred, MAT_deformed = self(volume_data)
        self.validation_outputs.append([voxel_pred, MAT_deformed])
        ce_loss, loss_dice, loss_sample, loss_point2sphere, loss_radius = self.loss(voxel_pred, MAT_deformed, batch)
        loss = loss_sample + self.config.lambda_p2s * loss_point2sphere + self.config.lambda_radius * loss_radius + self.config.lambda_ce * ce_loss + self.config.lambda_dice * loss_dice
        self.validation_losses.append([
            loss.item(),
            loss_sample.item(),
            loss_point2sphere.item(),
            loss_radius.item(),
            ce_loss.item(),
            loss_dice.item()
        ])

        target_segmentation = batch['label'].squeeze(0).long()
        shape_xyz = batch['surface_vtx'][0]
        self.validation_targets.append([target_segmentation, shape_xyz])

    def on_validation_epoch_end(self):
        loss_mean, loss_sample_mean, loss_point2sphere_mean, loss_radius_mean, ce_loss_mean, dice_loss_mean = np.mean(
            self.validation_losses, axis=0)
        values = {"val_loss": loss_mean, "val_sample": loss_sample_mean, "val_Point2Sphere": loss_point2sphere_mean,
                  "val_Radius": loss_radius_mean, "val_Cross Entropy": ce_loss_mean, "val_Dice Loss": dice_loss_mean}
        self.log_dict(values)

        for patient_pred, patient_target in zip(self.validation_outputs, self.validation_targets):
            pred_tensor = patient_pred[0]
            class_predictions = torch.argmax(F.softmax(pred_tensor, dim=1), dim=1)
            segmentation_mask = (class_predictions == 1).float()
            wandb.log({
                "Predicted": [wandb.Image(np.uint8(segmentation_mask[0][50].cpu().numpy() * 255))],
                "Target": [wandb.Image(np.uint8(patient_target[0][0][50].cpu().numpy() * 255))]
            })
            wandb.log({"val_3D Target": wandb.Object3D({
                "type": "lidar/beta",
                "points": patient_target[1].squeeze(0).cpu().numpy(),
            })})
            wandb.log({"val_3D Prediction": wandb.Object3D({
                "type": "lidar/beta",
                "points": (patient_pred[1][-1].squeeze())[:, :3].detach().cpu().numpy()
            })})

        self.validation_outputs.clear()
        self.validation_losses.clear()
        self.validation_targets.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config.learning_rate)
        return optimizer



class UNetLayer(nn.Module):
    """ U-Net Layer """
    def __init__(self, in_channels, out_channels):
        super(UNetLayer, self).__init__()

        conv1 = nn.Conv3d(in_channels,  out_channels, kernel_size=3, padding=1)
        conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        bn1 = nn.BatchNorm3d(out_channels)
        bn2 = nn.BatchNorm3d(out_channels)

        self.unet_layer = nn.Sequential(conv1, bn1, nn.ReLU(), conv2, bn2, nn.ReLU())

    def forward(self, x):
        return self.unet_layer(x)


class UNetModule(nn.Module):
    """ U-Net Module """
    def __init__(self, config):
        super(UNetModule, self).__init__()

        '''  Down layers '''
        self.max_pool = nn.MaxPool3d(2)
        self.down_layers = nn.ModuleList()
        self.down_layers.append(
            UNetLayer(
                in_channels=config.num_input_channels,
                out_channels=config.first_layer_channels
            )
        )
        for i in range(1, config.steps + 1):
            self.down_layers.append(
                UNetLayer(
                    in_channels=config.first_layer_channels * 2 ** (i - 1),
                    out_channels=config.first_layer_channels * 2 ** i
                )
            )

        '''  Up layers '''
        self.grid_upconv_layer = nn.ModuleList()
        self.grid_unet_layer = nn.ModuleList()
        for i in range(1, config.steps + 1):
            self.grid_upconv_layer.append(
                nn.ConvTranspose3d(
                    in_channels=config.first_layer_channels * 2 ** (config.steps - i + 1),
                    out_channels=config.first_layer_channels * 2 ** (config.steps - i),
                    kernel_size=2,
                    stride=2
                )
            )
            self.grid_unet_layer.append(
                UNetLayer(
                    in_channels=config.first_layer_channels * 2 ** (config.steps - i + 1),
                    out_channels=config.first_layer_channels * 2 ** (config.steps - i)
                )
            )

        ''' Final layer (for voxel decoder)'''
        self.final_layer = nn.Conv3d(
            in_channels=config.first_layer_channels,
            out_channels=config.num_classes,
            kernel_size=1
        )

    def forward(self, x):
        # first layer
        x = self.down_layers[0](x)
        down_outputs = [x]
        # down layers
        for unet_layer in self.down_layers[1:]:
            x = self.max_pool(x)
            x = unet_layer(x)
            down_outputs.append(x)

        decoder_features = [x]
        down_outputs.pop()
        for grid_upconv_layer, grid_unet_layer in zip(self.grid_upconv_layer, self.grid_unet_layer):
            x = grid_upconv_layer(x)
            x = crop_and_merge(down_outputs.pop(), x)
            x = grid_unet_layer(x)
            decoder_features.append(x)

        voxel_pred = self.final_layer(x)

        return voxel_pred, decoder_features


class MATDeform(nn.Module):
    """ MATDeform  """
    def __init__(self, config, skip_amount, A, D, latent_features_count, previous_latent_features_count=0):
        super(MATDeform, self).__init__()

        self.skip_amount = skip_amount
        self.A = A
        self.D = D
        self.skip_connection = LearntNeighbourhoodSampling(config, self.skip_amount)
        self.feature2feature = Features2Features(
            self.skip_amount + previous_latent_features_count + 4,
            latent_features_count,
            hidden_layer_count=config.graph_conv_layer_count
        )
        self.feature2vertex = Feature2VertexLayer(latent_features_count, 3)

    def forward(self, vertices, voxel_decoder_features, latent_features=None):
        skipped_features = self.skip_connection(voxel_decoder_features[:, :self.skip_amount], vertices)

        if latent_features is not None:
            latent_features = torch.cat([latent_features, skipped_features, vertices], dim=2)
        else:
            latent_features = torch.cat([skipped_features, vertices], dim=2)

        latent_features = self.feature2feature(latent_features, self.A, self.D)

        deltaV = self.feature2vertex(latent_features, self.A, self.D)
        vertices = vertices + deltaV

        return vertices, latent_features


class MATDecoder(nn.Module):
    """ MATDecoder  """
    def __init__(self, config, A, D):
        super(MATDecoder, self).__init__()

        self.skip_count = []
        self.latent_features_count = []
        self.MATDeformBlock = nn.ModuleList()
        for i in range(config.steps + 1):
            self.skip_count += [config.first_layer_channels * 2 ** (config.steps - i)]
            self.latent_features_count += [32]
            if i == 0:
                self.MATDeformBlock.append(MATDeform(config, self.skip_count[i], A, D, self.latent_features_count[i]))
            else:
                self.MATDeformBlock.append(MATDeform(config, self.skip_count[i], A, D, self.latent_features_count[i], self.latent_features_count[i-1]))


    def forward(self, vertices, decoder_features):
        MAT_deformed = []
        latent_features = None
        for MATDeformBlock, voxel_decoder_features in zip(self.MATDeformBlock, decoder_features):
            vertices, latent_features = MATDeformBlock(vertices, voxel_decoder_features, latent_features)
            MAT_deformed.append(vertices)
        return MAT_deformed


def crop_slices(shape1, shape2):
    slices = [slice((sh1 - sh2) // 2, (sh1 - sh2) // 2 + sh2) for sh1, sh2 in zip(shape1, shape2)]
    return slices


def crop_and_merge(tensor1, tensor2):
    slices = crop_slices(tensor1.size(), tensor2.size())
    slices[0] = slice(None)
    slices[1] = slice(None)
    slices = tuple(slices)
    return torch.cat((tensor1[slices], tensor2), 1)

def first_moment(mask):
    """
    Calculates the first moment of a given mask tensor.
    Parameters:
        mask (Tensor): A binary mask tensor.
    Returns:
        Tensor: The centroid of the mask tensor as a 3D tensor.
    """
    grid = torch.meshgrid(*[torch.arange(size).to(mask.device) for size in mask.shape])
    total_mass = mask.sum()
    epsilon = 1e-5
    centroid_x = ((grid[0] * mask).sum()) / (total_mass + epsilon)
    centroid_y = ((grid[1] * mask).sum()) / (total_mass + epsilon)
    centroid_z = ((grid[2] * mask).sum()) / (total_mass + epsilon)
    centroid = torch.stack([centroid_x, centroid_y, centroid_z])
    return centroid


