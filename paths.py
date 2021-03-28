import os


# Simple class to store all useful paths
class Paths:

    def __init__(self, db, model_ckp="") -> None:
        super().__init__()

        # the only 2 string to modify:
        self.db_name = db  # only this to change DB to use
        self.model_ckt = model_ckp

        self.dir_root = '/home/roncax/Git/Pytorch-UNet'

        self.dir_checkpoint = f'{self.dir_root}/data/checkpoints'
        self.dir_tensorboard_runs = f'{self.dir_root}/data/runs'

        # Database to analyze
        # raw
        self.dir_database = f'{self.dir_root}/data/databases/{self.db_name}'
        self.dir_raw_db = f'{self.dir_database}/raw_data'
        self.dir_processed_db = f'{self.dir_database}/processed_data'
        self.json_file = f'{self.dir_database}/{self.db_name}.json'

        # train
        self.dir_train_imgs = f'{self.dir_processed_db}/train/images'
        self.dir_train_masks = f'{self.dir_processed_db}/train/masks'

        # test
        self.dir_test = f'{self.dir_processed_db}/tests'
        self.dir_predicted_gifs = f'{self.dir_test}/gifs'
        self.dir_test_img = f'{self.dir_test}/img'
        self.dir_test_GTimg = f'{self.dir_test}/img_gt'
        self.dir_mask_prediction = f'{self.dir_test}/mask_prediction'
        self.dir_plot_saves = f'{self.dir_test}/plt_save'
        self.dir_plot_metrics = f'{self.dir_test}/metrics'

        ## Model
        self.dir_pretrained_model = f'{self.dir_checkpoint}/{self.model_ckt}'

        os.makedirs(self.dir_predicted_gifs, exist_ok=True)
