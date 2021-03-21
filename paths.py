from utilities.various import check_create_dir

dir_root = '/home/roncax/Git/Pytorch-UNet'

dir_checkpoint = f'{dir_root}/data/checkpoints'
dir_tensorboard_runs = f'{dir_root}/data/runs'

dir_raw_data = f'{dir_root}/data/databases/raw_data'
dir_processed_data = f'{dir_root}/data/databases/processed_data'

# Database to analyze
# raw
db_name = 'StructSeg2019_Task3_Thoracic_OAR'
dir_raw_db = f'{dir_raw_data}/{db_name}'

# train
dir_db = f'{dir_processed_data}/{db_name}'
dir_decompressed_imgs = f'{dir_db}/train'
dir_train_imgs = f'{dir_decompressed_imgs}/images'
dir_train_masks = f'{dir_decompressed_imgs}/masks'

# test
dir_test = f'{dir_db}/tests'
dir_predicted_gifs = f'{dir_test}/gifs'
dir_test_img = f'{dir_test}/img'
dir_test_GTimg = f'{dir_test}/img_gt'
dir_mask_prediction = f'{dir_test}/mask_prediction'
dir_plot_saves = f'{dir_test}/plt_save'

## Model
dir_pretrained_model = f'{dir_checkpoint}/2021-03-18 12:39:45.159008_CP_EPOCH0-LR(0.0001)_BS(1)_SCALE(1)_EPOCHS(1_VAL_LOSS(520.9430311918259).pth'


check_create_dir(dir_checkpoint)
check_create_dir(dir_tensorboard_runs)
check_create_dir(dir_processed_data)
check_create_dir(dir_db)
check_create_dir(dir_decompressed_imgs)
check_create_dir(dir_train_imgs)
check_create_dir(dir_train_masks)
check_create_dir(dir_test)
check_create_dir(dir_predicted_gifs)
check_create_dir(dir_test_img)
check_create_dir(dir_test_GTimg)
check_create_dir(dir_mask_prediction)
check_create_dir(dir_plot_saves)