from glob import glob
from typing import Mapping, Hashable
import os
from monai.config import KeysCollection
from monai.data import PersistentDataset, DataLoader, partition_dataset
import open3d as o3d
from monai.transforms import MapTransform
import torch
import numpy as np


class HeartDataset(PersistentDataset):
    def __init__(self, cfg, splits, transform=None):
        """
        Initializes an instance of the class.

        Parameters:
            cfg (Config): The configuration object.
            splits (list): A list of splits.
            transform (optional): An optional transform to be applied to the data.

        Returns:
            None
        """
        self.cfg = cfg
        subjects_dict = self._get_subjects_dict(splits)
        super().__init__(data=subjects_dict, transform=transform, cache_dir=cfg.dataset_cache)

    def _get_subjects_dict(self, splits):
        """
        Retrieves a dictionary of subjects from the Heart dataset.

        Returns:
            dict: A dictionary of subjects from the dataset.
        """
        images = sorted(glob(os.path.join(self.cfg.dataset_path, '*', '*', '*.nrrd')))
        labels = sorted(glob(os.path.join(self.cfg.dataset_path, '*', '*', '*.seg.nrrd')))
        images = [image for image in images if image not in labels]
        models = sorted(glob(os.path.join(self.cfg.dataset_path, '*', '*', '*.obj')))
        datalist = [{"image": image_name, "label": label_name, "surface_vtx": model_name} for image_name, label_name, model_name in zip(images, labels, models)]
        train_datalist, val_datalist = partition_dataset(
            datalist,
            ratios=[self.cfg.split, (1 - self.cfg.split)],
            shuffle=self.cfg.dataset_suffle,
            seed=self.cfg.seed
        )
        if splits == 'train':
            return train_datalist
        elif splits == 'val':
            return val_datalist
        else:
            raise Exception("Invalid split!!!")

    def get_loader(self):
        """
        Initializes a data loader object.
        Returns:
            DataLoader: The data loader object.
        """
        loader = DataLoader(self, batch_size=self.cfg.batch_size, num_workers=0, pin_memory=True)
        return loader


class LoadObjd(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            allow_missing_keys: bool = False
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                root = data[key]
                mesh = o3d.io.read_triangle_mesh(root)
                verts = torch.tensor(np.asarray(mesh.vertices))
                data[key] = verts.unsqueeze(0)
            else:
                raise ValueError(f"Key '{key}' not found in the data.")
        return data