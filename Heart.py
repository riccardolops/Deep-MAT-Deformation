import json
import logging
import logging.config
from pathlib import Path
import os
import numpy as np
import torchio as tio
from torch.utils.data import DataLoader
import torch
import open3d as o3d


class Heart(tio.SubjectsDataset):
    def __init__(self, root, splits, transform=None, **kwargs):
        root = Path(root)
        if not isinstance(splits, list):
            splits = [splits]

        subjects_list = self._get_subjects_list(root, splits)
        super().__init__(subjects_list, transform, **kwargs)

    def _get_subjects_list(self, root, splits):
        with open(os.path.join(root, 'dataset_new.json')) as dataset_file:
            json_dataset = json.load(dataset_file)

        subjects = []
        for split in splits:
            if split=='train':
                dataset = json_dataset[:10]
                for patient in dataset:
                    mesh = o3d.io.read_triangle_mesh(str(root / patient['surface']))
                    verts = torch.tensor(np.asarray(mesh.vertices))
                    subject_dict = {
                        'patient': patient,
                        'volume': tio.ScalarImage(root / patient['image']),
                        'segmentation': tio.LabelMap(root / patient['label']),
                        'surface_vtx': verts.unsqueeze(0),
                    }
                    subjects.append(tio.Subject(**subject_dict))
                print(f"Loaded {len(subjects)} patients for split {split}")
            elif split == 'val':
                dataset = json_dataset[-10:]
                for patient in dataset:
                    mesh = o3d.io.read_triangle_mesh(str(root / patient['surface']))
                    verts = torch.tensor(np.asarray(mesh.vertices))
                    subject_dict = {
                        'patient': patient,
                        'volume': tio.ScalarImage(root / patient['image']),
                        'segmentation': tio.LabelMap(root / patient['label']),
                        'surface_vtx': verts.unsqueeze(0),
                    }
                    subjects.append(tio.Subject(**subject_dict))
                print(f"Loaded {len(subjects)} patients for split {split}")
            else:
                logging.error("Dataset '{}' does not exist".format(split))
                raise SystemExit
        return subjects

    def get_loader(self, config):
        loader = DataLoader(self, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True)
        return loader
