import os


# Simple class to store all useful paths
class Paths:
    def __init__(self, db, model_ckp="") -> None:
        super().__init__()

        # Variables:
        self.db_name = db

        self.dir_root = '/home/roncax/Git/Pytorch-UNet'

        self.dir_checkpoint = f'{self.dir_root}/data/checkpoints'
        self.dir_tensorboard_runs = f'{self.dir_root}/data/runs'

        # raw
        self.dir_database = f'{self.dir_root}/data/databases/{self.db_name}'
        self.dir_raw_db = f'{self.dir_database}/raw_data'
        self.json_file = f'{self.dir_database}/{self.db_name}.json'

        # test
        self.dir_prediction = f'{self.dir_database}/prediction'
        self.dir_plots = f'{self.dir_database}/plots'

        self.dir_pretrained_model = f'{self.dir_checkpoint}/{model_ckp}'
        self.hdf5_db = f"{self.dir_database}/{self.db_name}.hdf5"
        self.hdf5_results = f"{self.dir_database}/{self.db_name}_predictions.hdf5"