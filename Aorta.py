import torchio as tio
from torch.utils.data import DataLoader
import os


class Aorta(tio.SubjectsDataset):
    def __init__(self, root, transform=None, **kwargs):
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
                lab = tio.LabelMap(label_path)
                subject_dict = {
                    'path': os.path.join(root, division, patient, patient.split(' ')[0]),
                    'volume': tio.ScalarImage(image_path, affine = lab.affine),
                    'segmentation': tio.LabelMap(label_path, affine = lab.affine)
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
