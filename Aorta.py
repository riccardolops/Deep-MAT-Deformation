from monai.data import Dataset, DataLoader, partition_dataset
import os
import numpy as np


class Aorta(Dataset):
    def __init__(self, root, transform=None, **kwargs):

        train_datalist, val_datalist = partition_dataset(
            datalist,
            ratios=[args.split, (1 - args.split)],
            shuffle=True,
            seed=args.seed,
        )

        subjects_list = self._get_subjects_list(root)


        super().__init__(subjects_list, transform, **kwargs)

    def _get_subjects_list(self, root):
        """
        Retrieves a list of subjects from the Aorta dataset.

        Parameters:
            root (str): The root directory of the dataset.

        Returns:
            list: A list of subjects from the dataset.
        """
        subjects = []
        divisions = os.listdir(root)
        for division in divisions:
            patients = os.listdir(os.path.join(root, division))
            for patient in patients:
                image_path = os.path.join(root, division, patient, patient.split(' ')[0] + '.nrrd')
                label_path = os.path.join(root, division, patient, patient.split(' ')[0] + '.seg.nrrd')
                volume = canonical(tio.ScalarImage(image_path))
                segment = canonical(tio.LabelMap(label_path))
                print(segment.affine == volume.affine)

                subject_dict = {
                    'path': os.path.join(root, division, patient, patient.split(' ')[0]),
                    'volume': volume,
                    'segmentation': segment
                }

                subjects.append(tio.Subject(**subject_dict))
        return subjects

    def get_loader(self, config):
        """
        Initializes a data loader object.

        Args:
            config (Config): The configuration object containing the batch size, number of workers, and whether to pin memory.

        Returns:
            DataLoader: The data loader object.

        """
        loader = DataLoader(self, batch_size=config.batch_size, num_workers=0, pin_memory=True)
        return loader
