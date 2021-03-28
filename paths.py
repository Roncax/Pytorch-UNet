import os

dir_root = '/home/roncax/Git/Pytorch-UNet'

dir_checkpoint = f'{dir_root}/data/checkpoints'
dir_tensorboard_runs = f'{dir_root}/data/runs'

#the only 2 string to modify:
db_name = "StructSeg2019_Task3_Thoracic_OAR"  # only this to change DB to use
model_ckpt = "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Model(Classic Unet)_Experiment(2)_Epoch(20).pth"

# Database to analyze
# raw
dir_database = f'{dir_root}/data/databases/{db_name}'
dir_raw_db = f'{dir_database}/raw_data'
dir_processed_db = f'{dir_database}/processed_data'
json_file = f'{dir_database}/{db_name}.json'

# train
dir_train_imgs = f'{dir_processed_db}/train/images'
dir_train_masks = f'{dir_processed_db}/train/masks'

# test
dir_test = f'{dir_processed_db}/tests'
dir_predicted_gifs = f'{dir_test}/gifs'
dir_test_img = f'{dir_test}/img'
dir_test_GTimg = f'{dir_test}/img_gt'
dir_mask_prediction = f'{dir_test}/mask_prediction'
dir_plot_saves = f'{dir_test}/plt_save'
dir_plot_metrics = f'{dir_test}/metrics'

## Model
dir_pretrained_model = f'{dir_checkpoint}/{model_ckpt}'

os.makedirs(dir_predicted_gifs, exist_ok=True)