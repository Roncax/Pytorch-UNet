import json
import logging
import argparse

from mod_unet.network_architecture.net_factory import build_net
from mod_unet.training.custom_trainer import CustomTrainer
from mod_unet.utilities.paths import Paths


def run_training(args):

    model = "SE-ResUnet"  # args.network
    db_name = "StructSeg2019_Task3_Thoracic_OAR"  # args.db
    epochs = args.epochs
    batch_size = 4  # args.batch_size
    lr = args.learning_rate
    val = args.validation_size
    patience = args.patience
    fine_tuning = args.fine_tuning
    feature_extraction = args.feature_extraction
    augmentation = False  # args.augmentation
    train_type = 'fine'  # args.train_type
    deep_supervision = False  # args.deep_supervision
    dropout = False  # args.dropout
    scale = args.scale
    fold = 5

    old_classes = args.old_classes
    paths = Paths(db=db_name)

    channels = 1

    labels = {"0": "Bg",
              "1": "RightLung",
              "2": "LeftLung",
              "3": "Heart",
              "4": "Trachea",
              "5": "Esophagus",
              "6": "SpinalCord"
              }  # dict_db_parameters["labels"]
    n_classes = 1 if len(labels) == 2 else len(labels)  # class number in net -> #classes+1(Bg)

    load_dir_list = {
        "1": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Experiment(664)_Epoch(23).pth",
        "2": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Experiment(664)_Epoch(23).pth",
        "3": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Experiment(664)_Epoch(23).pth",
        "4": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Experiment(664)_Epoch(23).pth",
        "5": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Experiment(664)_Epoch(23).pth",
        "6": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Experiment(664)_Epoch(23).pth",
        "coarse": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Experiment(664)_Epoch(23).pth"
    }

    # dice, bce, binaryFocal, multiclassFocal, crossentropy, dc_bce
    loss_criteria = {"1": "dc_bce",
                     "2": "dc_bce",
                     "3": "dc_bce",
                     "4": "dc_bce",
                     "5": "dc_bce",
                     "6": "dc_bce",
                     "coarse": "multiclassFocal"
                     }

    if train_type == "coarse":

        paths.set_pretrained_model(load_dir_list["coarse"])

        net = build_net(model=model, n_classes=n_classes, finetuning=fine_tuning, load_dir=paths.dir_pretrained_model,
                        channels=channels, old_classes=old_classes, feature_extraction=feature_extraction,
                        dropout=dropout, deep_supervision=deep_supervision)

        trainer = CustomTrainer(fold=fold, paths=paths, image_scale=scale, augmentation=augmentation,
                                batch_size=batch_size, loss_criterion=loss_criteria['coarse'], val_percent=val,
                                labels=labels, network=net, deep_supervision=deep_supervision)

        trainer.initialize()
        trainer.run_training()

    elif train_type == "fine":

        labels_list = filter(lambda x: x != '0', list(labels.keys()))

        for label in labels_list:
            dict_db_parameters = json.load(open(paths.json_file_database))
            dict_db_parameters["experiments"] += 1
            json.dump(dict_db_parameters, open(paths.json_file_database, "w"))
            paths.set_experiment_number(dict_db_parameters["experiments"])
            dict_results[dict_db_parameters["experiments"]] = {}
            dict_results[dict_db_parameters["experiments"]].update(temp_dict)
            dict_results[dict_db_parameters["experiments"]]["epochs"] = {}

            label_dict = {label: labels[label]}
            paths.dir_pretrained_model = load_dir_list[label]

            net = build_net(model=model,
                            n_classes=1,
                            finetuning=fine_tuning,
                            load_dir=paths.dir_pretrained_model,
                            device=device,
                            data_shape=data_shape, old_classes=old_classes, feature_extraction=feature_extraction,
                            verbose=verbose, dropout=dropout, deep_supervision=deep_supervision)

            dict_results[dict_db_parameters["experiments"]].update({"organ": label_dict[label]})



            trainer = CustomTrainer(weights=weight, fold=5, paths=paths, image_scale=scale, augmentation=augmentation,
                                    batch_size=batch_size, loss_criterion=loss_criteria[label], val_percent=val, labels=label_dict,
                                    dict_db_parameters=dict_db_parameters, network=net, deep_supervision=deep_supervision, device=device)
            trainer.initialize()
            trainer.run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--network", help="Unet, SE-ResUnet", required=True)
    parser.add_argument("--database_name", "-db", help="Supports: StructSeg2019_Task3_Thoracic_OAR, ...", required=True)
    parser.add_argument("--deterministic", "-det",
                        help="Makes training deterministic, but reduces training speed substantially. "
                             "Deterministic training will make you overfit to some random seed. "
                             "Don't use that.",
                        required=False, default=False)
    parser.add_argument("--epochs", "-e", required=False, default=1000)
    parser.add_argument("--batch_size", "-bs", required=False, default=1)
    parser.add_argument("--learning_rate", "-lr", required=False, default=0.0001)
    parser.add_argument("--scale", help="Downscaling factor of the images", required=False, default=1)
    parser.add_argument("--validation_size", "-val", help="% of the database that is used as validation (0-1)",
                        required=False, default=0.2)
    parser.add_argument("--patience",
                        help="Epochs patience of the early stopping. -1 means no early stopping (save all the epochs)",
                        required=False, default=5)
    parser.add_argument("--fine_tuning", help="Enable finetuning for every net (initialize the net before training)",
                        required=False, default=False)
    parser.add_argument("--feature_extraction", "-fx",
                        help="Enable feature extraction on every net (freeze all except last out layer)",
                        required=False, default=False)
    parser.add_argument("--verbose", "-v", required=False, default=True)
    parser.add_argument("--augmentation", "-aug", required=False, default=True)
    parser.add_argument("--train_type", required=False, default="multiclass", help="multibinary or multiclass")
    parser.add_argument("--deep_supervision", required=False, default=False)
    parser.add_argument("--dropout", required=False, default=False)
    parser.add_argument("--old_classes", required=False, default=0, help="Needed only if fine_tuning or "
                                                                         "feature_extraction, the model need to know "
                                                                         "the last param number of the final layer")
    parser.add_argument("--debug_mode", required=False, default=False,
                        help="Active debug mode if you not want to had permanent effect (e.g. save pth or epoch losses)")
    args = parser.parse_args()

    assert not (args.feature_extraction and args.fine_tuning), "Finetuning and feature extraction cannot be both active"
    if args.feature_extraction or args.fine_tuning: assert args.old_classes > 0, "Old classes needed to be specified"


    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    run_training(args)
