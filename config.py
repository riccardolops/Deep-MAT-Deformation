import os


class Config:
    def __init__(self):
        super(Config, self).__init__()
        self.trial_id = None
        self.save_dir_prefix = 'Experiment_'  # prefix for experiment folder
        self.name = 'voxel2mesh'
        self.save_path = None
        self.dataset_path = None

        self.num_workers = 12
        self.batch_size = 1
        # self.patch_shape = (100, 100, 100)
        self.grid_overlap = 0
        self.resize_shape = (96, 96, 96)
        self.skl_path = "./model/left_atrium_100_pt.ma"
        self.restore_ckpt = False
        self.save_path = "/home/rick/Documenti/Projects/models_waights"  # UPDATE HERE <<<<<<<<<<<<<<<<<<<<<<
        self.dataset_path = "/home/rick/Documenti/Projects/datasets/Task02_Heart"  # UPDATE HERE <<<<<<<<<<<<<<<<<<<<<<

        # Initialize data object
        assert self.save_path is not None, "Set cfg.save_path in config.py"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        assert self.dataset_path is not None, "Set cfg.dataset_path in config.py"

        self.profiler = None
        # "simple"
        ''' Dataset '''
        # input should be cubic. Otherwise, input should be padded accordingly.
        self.patch_shape = (64, 64, 64)

        self.ndims = 3
        self.augmentation_shift_range = 10

        ''' Model '''
        self.first_layer_channels = 16
        self.num_input_channels = 1
        self.steps = 4

        # Only supports batch size 1 at the moment.
        # self.batch_size = 1

        self.num_classes = 2
        self.batch_norm = True
        self.graph_conv_layer_count = 4

        ''' Optimizer '''
        self.learning_rate = 1e-3

        ''' Training '''
        self.numb_of_epochs = 50
        self.eval_every = 1  # saves results to disk

        ''' Pesi Loss '''
        self.lambda_p2s = 0.8
        self.lambda_radius = 0.1
        self.lambda_ce = 50
        self.lambda_dice = 50

        # ''' Rreporting '''
        # cfg.wab = True # use weight and biases for reporting
