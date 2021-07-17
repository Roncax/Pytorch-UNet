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
        self.dir_database = f'{self.dir_root}/data/{self.db_name}'
        self.dir_raw_db = f'{self.dir_database}/raw_data'
        self.json_file_database = f'{self.dir_database}/{self.db_name}.json'

        # test
        self.dir_prediction = f'{self.dir_database}/prediction'
        self.dir_plots = f'{self.dir_database}/plots'
        self.dir_stacking = f"{self.dir_root}/data/stacking"

        self.hdf5_db = f"{self.dir_database}/{self.db_name}.hdf5"
        self.hdf5_results = f"{self.dir_database}/{self.db_name}_predictions.hdf5"
        self.hdf5_stacking = f"{self.dir_stacking}/temp_stacking_db.hdf5"

        self.json_file_train_results = f"{self.dir_database}/train_results_{platform}.json"
        self.json_file_inference_results = f"{self.dir_database}/inference_results.json"
        self.json_stacking_experiments_results = f"{self.dir_stacking}/stacking_experiments.json"

    def set_experiment_number(self, n):
        self.dir_checkpoint = f'{self.dir_database}/checkpoints/{n}'
        
    def set_experiment_stacking_number(self, n):
        self.dir_checkpoint = f'{self.dir_stacking}/{n}'

    def set_train_stacking_results(self):
        self.json_file_train_results = self.json_stacking_experiments_results

    
    def set_pretrained_model_stacking(self, dir):
        self.dir_pretrained_model = f'{self.dir_root}/data/stacking/{dir}'


    def set_pretrained_model(self, dir):
        self.dir_pretrained_model = f'{self.dir_root}/data/checkpoints/{dir}'

