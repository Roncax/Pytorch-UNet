import json

import torch
from numpy import mean
from torch import optim
from torch.utils.data import DataLoader, random_split

from mod_unet.datasets.hdf5Dataset import HDF5Dataset
from mod_unet.training.loss_factory import build_loss
from mod_unet.training.network_trainer import NetworkTrainer
from mod_unet.utilities.tensorboard import Board
from mod_unet.evaluation import eval


class CustomTrainer(NetworkTrainer):
    def __init__(self, fold, paths, image_scale, augmentation, batch_size
                 , loss_criterion, val_percent, labels, network, deep_supervision, deterministic=False, fp16=True):
        super(CustomTrainer, self).__init__(deterministic, fp16)

        self.paths = paths
        self.network = network

        w = [float(x) / 100 for x in json.load(open(self.paths.json_file_database))["weights"].values()]
        self.weights = (
            torch.FloatTensor(w).to(device=self.device).unsqueeze(dim=0) if self.network.n_classes > 2 else None)

        self.output_folder = paths.dir_checkpoint
        self.fold = fold
        self.loss = build_loss(loss_criterion=loss_criterion, weight=self.weights, deep_supervision=deep_supervision)
        self.dataset_directory = paths.dir_database

        self.img_scale = image_scale
        self.dict_db_parameters = None
        self.labels = labels
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.val_percent = val_percent

        self.experiment_number = None

        self.dict_results = json.load(open(paths.json_file_train_results))

        #todo sistemare organo corrente
        organ = "coarse" if len(labels) > 2 else labels.keys()
        self.organs = self.dict_results[self.experiment_number].update()

    def set_experiment_number(self):
        dict_db_parameters = json.load(open(self.paths.json_file_database))
        dict_db_parameters["experiments"] += 1
        self.experiment_number = dict_db_parameters["experiments"]
        json.dump(dict_db_parameters, open(self.paths.json_file_database, "w"))
        self.paths.set_experiment_number(dict_db_parameters["experiments"])

    def initialize(self, training=True):
        super(CustomTrainer, self).initialize(training)
        self.set_experiment_number()
        self.initialize_optimizer_and_scheduler()
        self.load_dataset()
        self.was_initialized = True

    def initialize_optimizer_and_scheduler(self):
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=0.0001, weight_decay=1e-8, momentum=0.9)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode='min', patience=2)

    def validate(self, *args, **kwargs):
        loss_custom_val = eval.eval_train(self.network, self.val_gen, self.device, deep_supervision=False)
        self.print_to_log_file(f"The validation loss for this epoch is {loss_custom_val}", also_print_to_console=True)

    def load_dataset(self):
        # DATASET split train/val
        dataset = HDF5Dataset(scale=self.img_scale, mode='train', db_info=self.dict_db_parameters, paths=self.paths,
                              labels=self.labels, augmentation=self.augmentation)

        n_val = int(len(dataset) * self.val_percent)
        n_train = len(dataset) - n_val
        train, val = random_split(dataset, [n_train, n_val])
        self.tr_gen = DataLoader(train, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True,
                                 drop_last=True)
        self.val_gen = DataLoader(val, batch_size=1, shuffle=False, num_workers=8, pin_memory=True,
                                  drop_last=True)

    def tensorboard_setup(self, ):
        self.tsboard = Board(dataset_parameters=self.dataset.db_info, path=self.paths.dir_tensorboard_runs)

    def save_tsboard(self, dict):
        pass

    def json_log_save(self):
        temp_dict = {
            "organ":self.organs,
            "model": self.network.name,
            "epochs": self.max_num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.lr_threshold,
            "validation_size": self.val_percent,
            "patience": self.patience,
            "feature_extraction": self.feature_extraction,
            "deep_supervision": self.deep_supervision,
            "dropout": self.dropout,
            "fine_tuning": self.fine_tuning,
            "augmentation": self.augmentation,
            "loss_criteria": self.loss_criteria,
            "pretrained_model": self.pretrained_model,
            "class_weights": self.weights
        }

        json.load(open(self.paths.json_file_database))[self.experiment_number] = {}
        json.load(open(self.paths.json_file_database))[self.experiment_number].update(temp_dict)
        json.load(open(self.paths.json_file_database))[self.experiment_number]["epochs"] = {}
        pass

    def update_json_train(self, loss_val, loss_list, epoch):
        temp_dict = {"validation_loss": loss_val, "avg_train_loss": mean(loss_list)}
        dict_results = json.load(open(self.paths.json_file_train_results))
        dict_results[self.experiment_number]["epochs"][str(self.epoch)] = {}
        dict_results[self.experiment_number]["epochs"][str(epoch)].update(temp_dict)
        json.dump(dict_results, open(self.paths.json_file_train_results, "w"))
