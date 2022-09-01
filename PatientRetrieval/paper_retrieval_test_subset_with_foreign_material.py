import torch.nn as nn
from torchvision import models
from fastai import layers
from fastai.torch_core import *
import pytorch_lightning as pl
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning import testers
from pytorch_metric_learning import miners, losses, reducers, distances, regularizers
import pandas as pd
import numpy as np
import os
import torch
import torch.utils.data as data
from torchvision import transforms as tv
from PIL import Image
import time
import argparse
import json


# the resNet50 class,  can be optionally pretrained
class extractorRes50(nn.Module):
    def __init__(self, transfer_learning=True):
        super(extractorRes50, self).__init__()

        self.extractor_net = models.resnet50(pretrained=transfer_learning)
        # remove classification layer and global pooling layer
        self.extractor_net = nn.Sequential(*(list(self.extractor_net.children())[:-2]))

    def forward(self, x):
        x = self.extractor_net(x)
        return x

    def freeze(self):
        for param in self.extractor_net.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.extractor_net.parameters():
            param.requires_grad = True


# our model head, used after extractor, here pooling and embedding size is defined
class model_head(nn.Module):
    def __init__(self, channel_size_in, embedding_size, pooling_size=1, channel_size_pre_flat=100):
        super(model_head, self).__init__()
        self.pooling = layers.AdaptiveConcatPool2d(pooling_size)
        self.bottle_neck_conv = nn.Conv2d(channel_size_in, channel_size_pre_flat, kernel_size=1, stride=1, bias=False)
        num_input_features = channel_size_pre_flat * (pooling_size ** 2) * 2
        bn1 = nn.BatchNorm1d(num_input_features)
        bn2 = nn.BatchNorm1d(2048)
        self.first_fc = nn.Sequential(nn.Flatten(),
                                      bn1,
                                      nn.Dropout(0.5),
                                      nn.Linear(num_input_features, 2048),
                                      nn.ReLU(True))
        self.second_fc = nn.Sequential(
            bn2,
            nn.Dropout(0.25),
            nn.Linear(2048, embedding_size))

    def forward(self, x):
        x = self.pooling(x)
        bs, chs, height, width = x.shape
        x = x.view((bs, int(chs / 2), height * 2, width))
        x = self.bottle_neck_conv(x)
        x = self.first_fc(x)
        return self.second_fc(x)


# wrapper for all network parts, in this case slightly altered/slimmed down for testing purposes
class complete_model_test(pl.LightningModule):
    def __init__(self,
                 extractor,
                 head,
                 loss_func,
                 learning_rate=1e-5,
                 batch_size=32,
                 expected_num_epochs=1,
                 weight_decay=1e-5,
                 num_workers=0):
        super().__init__()
        self.save_hyperparameters('learning_rate', 'batch_size', 'expected_num_epochs', 'weight_decay', 'num_workers')

        # tool to compute our metrics of interest, as well as the distances within the dataset
        self.distance = distances.LpDistance(normalize_embeddings=False)
        self.accuracy_calculator = AccuracyCalculator(include=("mean_average_precision_at_r", "precision_at_1",
                                                               "r_precision"), avg_of_avgs=False)
        self.loss_func = loss_func

        # combine all
        self.model = nn.Sequential(extractor, head)

    def forward(self, x):
        x = self.model(x)
        return x

    # exclude certain layers from weight decay, taken from
    # https://github.com/PyTorchLightning/PyTorch-Lightning-Bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py#L64-L331
    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=['bias', 'bn']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {'params': params, 'weight_decay': weight_decay},
            {'params': excluded_params, 'weight_decay': 0.}, ]

    def configure_optimizers(self):
        params_to_train = self.exclude_from_wt_decay(self.model.named_parameters(),
                                                     weight_decay=self.hparams.weight_decay)
        optimizer = torch.optim.SGD(params_to_train, lr=self.hparams.learning_rate, momentum=0.9,
                                    weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.learning_rate,
                                                        total_steps=(int(61755 / 32) + 1) * self.hparams.expected_num_epochs + 1,
                                                        epochs=None,
                                                        # overall 61755 images / batchsize 32 = 2165 = number of steps in the scheduler
                                                        steps_per_epoch=None,
                                                        pct_start=0.3, anneal_strategy='cos',
                                                        cycle_momentum=True, base_momentum=0.85,
                                                        max_momentum=0.95, div_factor=25.0,
                                                        final_div_factor=10000.0, last_epoch=-1)
        scheduler = {"scheduler": scheduler,
                     "interval": "step",
                     "frequency": 1}

        return [optimizer], [scheduler]

    # define diff process steps during training validation and testing, describing the data flow and logging
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        embeddings = self(inputs)
        loss = self.loss_func(embeddings, labels)
        mean_distance = torch.mean(self.distance.compute_mat(embeddings.type(torch.float), None))
        max_distance = torch.max(self.distance.compute_mat(embeddings.type(torch.float), None))
        metrics = {'train_loss': loss,
                   'mean_distance': mean_distance,
                   'max_distance': max_distance}
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        embeddings = self(inputs)
        loss = self.loss_func(embeddings, labels)
        metrics = {'val_loss': loss}
        self.log_dict(metrics)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        embeddings = self(inputs)
        loss = self.loss_func(embeddings, labels)
        self.log('test_loss', loss)

    # The Helper function to compute the embeddings from the while training set
    def get_all_embeddings(self, dataset, model):
        tester = testers.BaseTester(dataloader_num_workers=self.hparams.num_workers)
        return tester.get_all_embeddings(dataset, model)

    # the actual testing algorithm, done every epoch, but can also be called outside of training with any given testset
    # as a parameter
    def test_various_metrics(self, testset, save_path=None):
        t0 = time.time()
        device = torch.device("cuda")
        self.model = self.model.to(device)
        embeddings, labels = self.get_all_embeddings(testset, self.model)

        print("Computing accuracy")
        accuracies = self.accuracy_calculator.get_accuracy(embeddings, embeddings, np.squeeze(labels),
                                                           np.squeeze(labels), True)
        dist_mat = self.distance.compute_mat(torch.tensor(embeddings, dtype=torch.float), None)
        mean_distance = torch.mean(dist_mat)
        max_distance = torch.max(dist_mat)
        print("mean_distance = " + str(mean_distance))
        print("max_distance = " + str(max_distance))
        print("Test set accuracy (MAP@R) = {}".format(accuracies["mean_average_precision_at_r"]))
        print("r_prec = " + str(accuracies["r_precision"]))
        print("prec_at_1 = " + str(accuracies["precision_at_1"]))
        t1 = time.time()
        print("Time used for evaluating: " + str((t1 - t0) / 60) + " minutes")

        metrics = {"MAP@R_test": accuracies["mean_average_precision_at_r"],
                   "r_precision": accuracies["r_precision"],
                   "precision_at_1": accuracies["precision_at_1"],
                   "mean__val_distance": mean_distance,
                   "max_val_distance": max_distance
                   }

        if save_path:
            with open(save_path, 'a+') as f:
                f.write('MAP@R: %s\n' % accuracies["mean_average_precision_at_r"])
                f.write('mean_distance: %s\n' % str(mean_distance))
                f.write('max_distance: %s\n' % str(max_distance))
                f.write('MAP@R: %s\n' % accuracies["mean_average_precision_at_r"])
                f.write('R-Precision: %s\n' % str(accuracies["r_precision"]))
                f.write('Precision@1: %s\n' % accuracies["precision_at_1"])

        self.log_dict(metrics)


class eval_Dataset(data.Dataset):
    def __init__(self, root, csv_file_path, csv_singles=None, size=None):
        self.root = root
        patient_data_many = pd.read_csv(csv_file_path)
        name_array = patient_data_many['Image Index'].to_list()
        id_array = patient_data_many['Patient ID'].to_list()
        if csv_singles:
            patient_data_singles = pd.read_csv(csv_singles)
            name_array_singles = patient_data_singles['Image Index'].to_list()
            id_array_singles = patient_data_singles['Patient ID'].to_list()
            name_array = name_array + name_array_singles
            id_array = id_array + id_array_singles
        self.name_array = name_array
        self.id_array = id_array
        self.num_samples = len(self.name_array)

        if size is not None:
            transform_list = [
                tv.transforms.Resize(size, interpolation=2),
                tv.transforms.Resize([1024, 1024], interpolation=2),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        else:
            transform_list = [
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

        self.transform = tv.transforms.Compose(transform_list)

    def __getitem__(self, idx):
        # extract name and id
        name = self.name_array[idx]
        patient_id = self.id_array[idx]
        img = self.transform(Image.open(os.path.join(self.root, name)).convert('RGB'))
        return img, patient_id

    def __len__(self):
        return self.num_samples


def main():
    # define an argument parser
    parser = argparse.ArgumentParser('Patient Retrieval Test With Foreign Material')
    parser.add_argument('--config_path', default='./config_files/', help='the path where the config files are stored')
    parser.add_argument('--config', default='config.json',
                        help='the hyper-parameter configuration and experiment settings')
    args = parser.parse_args()
    print('Arguments:\n' + '--config_path: ' + args.config_path + '\n--config: ' + args.config)

    # read config
    with open(args.config_path + args.config, 'r') as config:
        config = config.read()

    # parse config
    config = json.loads(config)
    # here, once again, the paths to the csv files and image folder have to be set, as well as to the model to be tested
    csv_foreign_material = config["csv_foreign_material"]
    image_root = config["image_root"]
    dict_path = config["model_save_path"] + config["final_model_name"]
    # again, the amount of workers used by the CPU
    num_workers = config["num_workers"]

    # make training deterministic
    pl.seed_everything(3)
    # initialize Network
    # initialize model as a whole
    res50_extractor = extractorRes50()
    network_head = model_head(channel_size_in=2048, embedding_size=128, pooling_size=5)
    model50 = complete_model_test(res50_extractor, network_head, loss_func=None, num_workers=num_workers)
    model50.model.load_state_dict(torch.load(dict_path))

    # initialize Network
    test_set = eval_Dataset(image_root, csv_foreign_material, size=[1024, 1024])

    #  load the new state dict
    model50.test_various_metrics(test_set, config["foreign_data_save_path"])


if __name__ == '__main__':
    main()
