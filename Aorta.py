import torchio as tio
from pathlib import Path
from torch.utils.data import DataLoader



class Aorta(tio.SubjectsDataset):
    def __init__(self, root, splits, transform=None, **kwargs):
        root = Path(root)
        if not isinstance(splits, list):
            splits = [splits]

        subjects_list = self._get_subjects_list(root, splits)
        super().__init__(subjects_list, transform, **kwargs)

    def _get_subjects_list(self, root, splits):
        """
        Retrieves a list of subjects from the dataset.

        Parameters:
            root (str): The root directory of the dataset.
            splits (list): The splits to retrieve subjects for.

        Returns:
            list: A list of subjects from the dataset.
        """

        return subjects

    def get_loader(self, config):
        """
        Initializes a data loader object.

        Args:
            config (Config): The configuration object containing the batch size, number of workers, and whether to pin memory.

        Returns:
            DataLoader: The data loader object.

        """
        loader = DataLoader(self, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True)
        return loader
