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
import random
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms as tv
from PIL import Image
from pytorch_lightning import loggers as pl_loggers
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


# wrapper for all network parts, takes in extractor, head and the loss function, here, PyTorch lightning was used as a
# framework
class complete_model(pl.LightningModule):
    def __init__(self,
                 extractor,
                 head,
                 loss_func,
                 validation_path_many,
                 validation_path_singles,
                 root,
                 frozen_at_start=False,   # -> whether Extractor is frozen or not
                 learning_rate=1e-5,
                 batch_size=32,
                 expected_num_epochs=1,
                 weight_decay=1e-5,
                 num_workers=0):

        super().__init__()
        self.save_hyperparameters('frozen_at_start', 'learning_rate', 'batch_size', 'expected_num_epochs',
                                  'weight_decay', 'num_workers')

        # tool to compute our metrics of interest, as well as the distances within the dataset
        self.distance = distances.LpDistance(normalize_embeddings=False)
        self.accuracy_calculator = AccuracyCalculator(include=("mean_average_precision_at_r", "precision_at_1",
                                                               "r_precision"), avg_of_avgs=False)
        self.loss_func = loss_func

        # used for val logging
        self.test_set_val = eval_Dataset(root, validation_path_many, validation_path_singles)

        if self.hparams.frozen_at_start:
            extractor.freeze()

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

        """ 
        This part is rather important, because you have to change the number of total steps when training, because the 
        learning rate depends on it when using OneCycleLR. Generally something like num_samples / batchsize
        """

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.learning_rate,
                                                        # num_samples/batchsize = 61755/32
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

    def validation_epoch_end(self, outs):
        # outs is not needed here, this part could be rather easily optimized so the embedding space doesn't have to be
        # computed twice
        loss = torch.stack(outs).mean()
        print("average_val_loss: " + str(loss))
        self.test_various_metrics(self.test_set_val)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        embeddings = self(inputs)
        loss = self.loss_func(embeddings, labels)
        self.log('test_loss', loss)

    # The Helper function to compute the embeddings from the while training set
    def get_all_embeddings(self, dataset, model):
        tester = testers.BaseTester(dataloader_num_workers=self.hparams.num_workers)
        return tester.get_all_embeddings(dataset, model)

    # the actual testing algorithm, done every epoch, but can also be called outside of training with any given test set
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
        print("r_prec = "+ str(accuracies["r_precision"]))
        print("prec_at_1 = "+ str(accuracies["precision_at_1"]))
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


# a simple function that takes in id and name list, and takes out the necessary amount of samples
def sample_func(current_id, num_samples, name_array, id_array):
    candidates = name_array[current_id == id_array]
    num_samples = int(num_samples)
    samples = np.random.choice(candidates, num_samples, replace=False)
    filter_mask = np.isin(name_array, samples, invert=True)
    id_array = id_array[filter_mask]
    name_array = name_array[filter_mask]
    return samples, name_array, id_array


# This is the part that does the heavy lifting of the mining list creation. This function forms blocks of size
# block_size, returns a boolean containing whether the operation was successful as well as the block and the updated id
# and name list. It generally uses a mix of heuristics to guarantee that a set of conditions are met. Of these, the most
# important one is, that images of classes are not unnecessarily split up and do not create singletons within a block,
# meaning that if there is a patient with more than one image, it should be guaranteed that those images are not split
# up in a way that there is a batch containing only a single one of these images instead of a group. To do so, the
# modulo operator is used as a basis, so the amount of images can always be seen as relative to the block_size that is
# used.
def get_block(block_size, name_array, id_array):
    operation_successful = True
    block = []
    # At the start, a random patient with multiple images is selected
    current_id = np.random.choice(id_array)
    ids, counts = np.unique(id_array, return_counts=True)
    current_count = counts[np.argwhere(ids == current_id)]

    # In the case that the amount of images exactly matches the block_size or is a multiple of it, block_size images are
    # used for the block and directly returned.
    if current_count % block_size == 0:
        samples, name_array, id_array = sample_func(current_id, block_size, name_array, id_array)
        for sample in samples:
            block.append((sample, current_id))
        return operation_successful, block, name_array, id_array

    # If there are more than block_size images with this id, different cases are considered. If the amount of images is
    # exactly one bigger than the block size, a pair of two images is collected from this class. When there are at least
    # block_size + 2 images, there is a 5050 chance whether we simply collect block_size images or another amount. This
    # other amount is normally current_count % block_size, so the rest that would remain after using the modulo
    # operation, with block_size as the basis on the amount of images we have of the patient. The only special case we
    # have here is the case where the resulting number would be block_size - 1, meaning that there would be exactly one
    # spot in the block left, automatically leading to a singleton. Therefore, we sample block_size - 2 samples in this
    # case.
    if current_count > block_size:
        # check whether its 9 or mod8 == 1
        if current_count % 8 == 1:
            samples, name_array, id_array = sample_func(current_id, 2, name_array, id_array)
            for sample in samples:
                block.append((sample, current_id))
        # 50/50 whether we fill the block with block_size or current_count % block_size
        else:
            if random.random() < 0.5:
                # fill with the block_size
                samples, name_array, id_array = sample_func(current_id, block_size, name_array, id_array)
                for sample in samples:
                    block.append((sample, current_id))
                return operation_successful, block, name_array, id_array
            else:
                # In case block_size - 1, there is not enough space for another pair, and we only take block_size - 2
                if current_count % block_size == block_size - 1:
                    samples, name_array, id_array = sample_func(current_id, current_count % block_size - 2, name_array,
                                                                id_array)
                    for sample in samples:
                        block.append((sample, current_id))
                # all the other cases
                else:
                    samples, name_array, id_array = sample_func(current_id, current_count % block_size, name_array,
                                                                id_array)
                    for sample in samples:
                        block.append((sample, current_id))
    else:
        samples, name_array, id_array = sample_func(current_id, current_count % block_size, name_array, id_array)
        for sample in samples:
            block.append((sample, current_id))

    # This part of the function describes the routine if the first sampling step was not enough to reach an amount of
    # block_size images. From here on, a loop is used to iteratively fill up the remaining space in the block. In the
    # first step, all patients whose mod rest is larger than one and those that would either completely fill the
    # remaining block or would leave a remaining block size that is bigger than 1 are considered. If there are none
    # left, we try to fill up the block with a patient that has a mod rest of size block_size - 1. If that is not
    # possible as well, the loop ends and the unfinished block is returned with the operation_successful flag set to
    # false.
    rest = block_size - len(block)

    while rest != 0:
        ids, counts = np.unique(id_array, return_counts=True)
        shortened_counts = np.mod(counts, block_size)
        # filter possible ids, so 1 < num_ids and rest - num_ids < 2
        possible_ids = ids[
            (shortened_counts > 1) & (((rest - shortened_counts) > 1) | ((rest - shortened_counts) == 0))]

        # no combinations available
        if possible_ids.size == 0:
            possible_ids = ids[shortened_counts == block_size - 1]
            # not even when splitting up block_size - 1 block
            if possible_ids.size == 0:
                operation_successful = False
            # fill up with a block_size - 1 block --> operation successful
            else:
                current_id = np.random.choice(possible_ids)
                samples, name_array, id_array = sample_func(current_id, rest, name_array, id_array)
                for sample in samples:
                    block.append((sample, current_id))
            return operation_successful, block, name_array, id_array
            # otherwise continue filling til rest == 0
        else:
            current_id = np.random.choice(possible_ids)
            samples, name_array, id_array = sample_func(current_id, shortened_counts[ids == current_id], name_array,
                                                        id_array)
            for sample in samples:
                block.append((sample, current_id))
            rest = block_size - len(block)
    # the loop ended, so we reached block_size 0 and were successful
    return operation_successful, block, name_array, id_array


# The listbatch creation workflow. The main idea was to create a sample list from the target csv that can then be
# directly iterated over when using a data loader. For doing so, patients are sampled in blocks which then get appended
# to the final list. The block_size determines the maximum number of images from a patient that can be sampled at the
# same time. The batch size should be a multiple of the block_size so the sampling works correctly. Generally this was
# mainly done with a block_size of 8 and a batch size of 32 in mind, but could potentially be altered. This could,
# however, lead to different results compared to the ones published in the paper.
#
#
# The higher level of the list creation. Here, blocks are step by step sampled and appended to the list used in the data
# loader. Blocks that cannot be completed successfully are collected and then later on added to the list once there are
# no more possibilities of creating a block.
def build_mining_list_32(name_array, id_array, block_size):
    mining_list = []
    garbage_collector = []
    while name_array.size != 0:
        operation_successful, block, name_array, id_array = get_block(block_size, name_array, id_array)

        if not operation_successful:
            for part in block:
                garbage_collector.append(part)
        else:
            for image_name in block:
                mining_list.append(image_name)

    if garbage_collector:
        for garbage in garbage_collector:
            mining_list.append(garbage)

    print("successfully created mining list")
    return mining_list


class miningDataset(data.Dataset):
    def __init__(self, root, csv_file_path, train, bucket_size=8):
        self.root = root
        self.train = train
        self.patient_data_frame = pd.read_csv(csv_file_path)
        self.name_array = self.patient_data_frame['Image Index'].to_numpy()
        self.id_array = self.patient_data_frame['Patient ID'].to_numpy()
        # create list that is ordered for our use case
        self.mining_list = build_mining_list_32(self.name_array, self.id_array, bucket_size)
        self.num_samples = len(self.mining_list)

        if self.train:
            transform_list = [
                tv.transforms.ColorJitter(brightness=0.4, saturation=0, contrast=0.4, hue=0),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        else:
            transform_list = [
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

        self.transform = tv.transforms.Compose(transform_list)

    def __getitem__(self, idx):
        # extract name and id
        name, patient_id = self.mining_list[idx]
        # open  and transform image
        img = self.transform(Image.open(os.path.join(self.root, name)).convert('RGB'))
        return img, patient_id

    def __len__(self):
        return self.num_samples


class eval_Dataset(data.Dataset):
    def __init__(self, root, csv_many, csv_singles, size=None):
        self.root = root
        patient_data_many = pd.read_csv(csv_many)
        name_array_many = patient_data_many['Image Index'].to_list()
        id_array_many = patient_data_many['Patient ID'].to_list()
        patient_data_singles = pd.read_csv(csv_singles)
        name_array_singles = patient_data_singles['Image Index'].to_list()
        id_array_singles = patient_data_singles['Patient ID'].to_list()
        self.name_array = name_array_many + name_array_singles
        self.id_array = id_array_many + id_array_singles
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


# works as a wrapper for all things related to the datasets/loaders, again PyTorch Lightning based
class MiningDataModule(pl.LightningDataModule):
    def __init__(self,
                 root_images,
                 path_train,
                 path_val,
                 path_test,
                 batch_size=32,
                 num_workers=0,
                 single_csv_val=None,
                 single_csv_test=None):
        super().__init__()

        self.single_csv_val = single_csv_val
        self.single_csv_test = single_csv_test
        self.root_images = root_images
        self.path_train = path_train
        self.path_val = path_val
        self.path_test = path_test
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        # split dataset
        if stage == 'fit':
            self.train_set = miningDataset(self.root_images, self.path_train, train=True)
            self.val_set = eval_Dataset(self.root_images, self.path_val, self.single_csv_val)
        if stage == 'test':
            self.test_set = eval_Dataset(self.root_images, self.path_test, self.single_csv_test)

    # return the dataloader for each split
    def train_dataloader(self):
        self.train_set = miningDataset(self.root_images, self.path_train, train=True)
        train_loader = DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)
        return train_loader

    def val_dataloader(self):
        self.val_set = eval_Dataset(self.root_images, self.path_val, self.single_csv_val)
        val_loader = DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)
        return test_loader


class loss_wrapper(torch.nn.Module):
    def __init__(self, loss_func, embedding_size=128, miner=None, cbm_size=None):
        super(loss_wrapper, self).__init__()
        self.embedding_size = embedding_size
        self.cbm_size = cbm_size
        self.miner = miner
        self.loss_func = loss_func
        if self.cbm_size is not None:
            self.loss_func = losses.CrossBatchMemory(self.loss_func, embedding_size, memory_size=cbm_size,
                                                     miner=self.miner)

    def forward(self, embeddings, labels):
        if self.miner:
            indices = self.miner(embeddings, labels)
            loss = self.loss_func(embeddings, labels, indices)
        else:
            loss = self.loss_func(embeddings, labels)
        return loss


def main():
    # define an argument parser
    parser = argparse.ArgumentParser('Patient Retrieval Phase1')
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

    # the paths to the image folder and the csv files
    image_root = config["image_root"]
    csv_train = config["csv_train"]
    csv_val = config["csv_val"]
    csv_test = config["csv_test"]
    csv_val_singles = config["csv_val_singles"]
    csv_test_singles = config["csv_test_singles"]
    # the path the intermediate model should be saved to
    model_save_path = config["model_save_path"] + config["intermediate_model_name"]

    # initializes the tensorboard logger, within the directory, logfiles and training checkpoints are created
    tb_logger = pl_loggers.TensorBoardLogger(
        config["log_path"], config["log_name_phase_1"]
    )

    '''Here, the hyper parameters can be set. The given ones are the ones we used for the results from the paper'''
    weight_decay = config["weight_decay"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    cbm_size = config["cbm_size"]

    # dependent on the CPU, when available, using a higher amount of workers will speed up the process
    num_workers = config["num_workers"]

    # the amount of epochs that we used for training only the head. Expected, max and min epochs should be set to the
    # same value
    expected_num_epochs = config["num_epochs_phase1"]
    max_epochs = config["num_epochs_phase1"]
    min_epochs = config["num_epochs_phase1"]

    # Parameter to decide whether the extractor should be frozen or not. In this first part it was in our case frozen
    frozen_at_start = config["extractor_frozen_phase1"]

    # make training deterministic
    pl.seed_everything(3)

    # initialize network
    res50_extractor = extractorRes50()
    network_head = model_head(channel_size_in=2048, embedding_size=128, pooling_size=5)

    # initialize loss
    loss_func = losses.ContrastiveLoss()
    loss = loss_wrapper(loss_func, embedding_size=128, cbm_size=cbm_size)

    # initialize model as a whole
    model50 = complete_model(res50_extractor, network_head, loss, frozen_at_start=frozen_at_start,
                             learning_rate=learning_rate, batch_size=batch_size,
                             expected_num_epochs=expected_num_epochs, weight_decay=weight_decay,
                             validation_path_many=csv_val, root=image_root, validation_path_singles=csv_val_singles,
                             num_workers=num_workers)

    # when loading an existing model, the two lines below can be commented in with the correct path
    # model_dict = '/path/to/model/dict'
    # model50.model.load_state_dict(torch.load(model_dict))

    trainer50 = pl.Trainer(
        weights_summary=None,
        reload_dataloaders_every_epoch=True,
        logger=tb_logger,
        progress_bar_refresh_rate=250,
        val_check_interval=1.0,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        precision=16,
        accumulate_grad_batches=1,
        deterministic=True,
        gpus=1
    )

    data_module = MiningDataModule(
        root_images=image_root,
        path_train=csv_train,
        path_val=csv_val,
        path_test=csv_test,
        batch_size=model50.hparams.batch_size,
        num_workers=num_workers,
        single_csv_val=csv_val_singles,
        single_csv_test=csv_test_singles
    )

    t1 = time.time()
    trainer50.fit(model50, data_module)
    t2 = time.time()
    print("Time used for fitting the model: " + str((t2-t1)/3600) + " hours")

    torch.save(model50.model.state_dict(), model_save_path)


if __name__ == '__main__':
    main()
