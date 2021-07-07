# Simple class to store all useful paths
class Paths:
    def __init__(self, db, platform) -> None:
        super().__init__()

        # Variable:
        self.db_name = db
        self.dir_pretrained_model = ''
        self.dir_checkpoint = ''


        if platform == "local":
            self.dir_root = '/home/roncax/Git/Pytorch-UNet'
        elif platform == "colab":
            self.dir_root = '/content/gdrive/MyDrive/Colab/Thesis_OaR_Segmentation'
        elif platform == "polimi":
            self.dir_root = ''
            

        # raw
        self.dir_database = f'{self.dir_root}/data/databases/{self.db_name}'
        self.dir_raw_db = f'{self.dir_database}/raw_data'
        self.json_file_database = f'{self.dir_database}/{self.db_name}.json'

        # test
        self.dir_prediction = f'{self.dir_database}/prediction'
        self.dir_plots = f'{self.dir_database}/plots'

        self.hdf5_db = f"{self.dir_database}/{self.db_name}.hdf5"
        self.hdf5_results = f"{self.dir_database}/{self.db_name}_predictions.hdf5"

        self.json_file_train_results = f"{self.dir_database}/train_results_{platform}.json"

        self.json_file_inference_results = f"{self.dir_database}/inference_results.json"

    def set_experiment_number(self, n):
        self.dir_checkpoint = f'{self.dir_root}/data/checkpoints/{n}'

    def set_pretrained_model(self, dir):
        self.dir_pretrained_model = f'{self.dir_root}/data/checkpoints/{dir}'

