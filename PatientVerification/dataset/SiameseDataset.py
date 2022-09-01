from torch.utils import data
import numpy as np
import random
from PIL import Image


class SiameseDataset(data.Dataset):
    def __init__(self, phase='training', data_handling='balanced', n_channels=3, n_samples=792294, transform=None,
                 image_path='./', save_path=None):

        self.phase = phase
        self.data_handling = data_handling

        if (self.data_handling == 'balanced' or self.data_handling == 'randomized') and n_samples > 792294:
            # maximum amount of image pairs (for training) in the balanced case is 792294
            raise ValueError('Invalid value for parameter n_samples! The maximum amount of image pairs is 792294.')
        else:
            self.n_samples = n_samples

        if self.phase == 'training':
            # To ensure patient-wise image splits, we split train_val_list.txt at index 75708.
            # In this way, images from one patient only appear in one subset.
            filenames = np.loadtxt('./train_val_list.txt', dtype=str)[:75708]
            self.image_pairs = np.loadtxt('./image_pairs/pairs_training.txt', dtype=str)
        elif self.phase == 'validation':
            filenames = np.loadtxt('./train_val_list.txt', dtype=str)[75708:]
            self.image_pairs = np.loadtxt('./image_pairs/pairs_validation.txt', dtype=str)
        elif self.phase == 'testing':
            filenames = np.loadtxt('./test_list.txt', dtype=str)
            self.image_pairs = np.loadtxt('./image_pairs/pairs_testing.txt', dtype=str)
        else:
            raise Exception('Invalid argument for parameter phase!')

        if self.data_handling == 'balanced' and self.n_samples < 792294:
            # pick the first n_samples/2 positive and negative pairs from the .txt file
            positive_pairs = self.image_pairs[:int(self.n_samples / 2)]
            negative_pairs = self.image_pairs[
                             int(len(self.image_pairs) / 2):int(len(self.image_pairs) / 2) + int(self.n_samples / 2)]
            self.image_pairs = np.concatenate((positive_pairs, negative_pairs), axis=0)

        elif self.data_handling == 'randomized' and self.n_samples <= 792294:
            N = int(self.n_samples / 2)
            i = 0
            neg_list = []

            while i < N:
                file1 = random.choice(filenames)
                file2 = random.choice(filenames)

                if file1[:-8] != file2[:-8]:
                    sample = [file1, file2, str(0.0)]
                    neg_list.append(sample)
                    i += 1

            neg_list = np.asarray(neg_list)

            positive_pairs = self.image_pairs[:int(self.n_samples / 2), :3]
            negative_pairs = np.asarray(random.sample(list(neg_list), int(self.n_samples / 2)))
            self.image_pairs = np.concatenate((positive_pairs, negative_pairs), axis=0)

        self.n_channels = n_channels
        self.transform = transform
        self.PATH = image_path

        if self.data_handling == 'balanced' and save_path is not None:
            f = open(save_path + 'image_pairs_' + self.phase + '.txt', 'w+')
            for i in range(len(self.image_pairs)):
                f.write(str(self.image_pairs[i][0]) + '\t' + str(self.image_pairs[i][1]) + '\t' +
                        str(self.image_pairs[i][2]) + '\n')
            f.close()

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):

        x1 = pil_loader(self.PATH + self.image_pairs[index][0], self.n_channels)
        x2 = pil_loader(self.PATH + self.image_pairs[index][1], self.n_channels)

        if self.transform is not None:
            x1 = self.transform(x1)
            x2 = self.transform(x2)

        y1 = float(self.image_pairs[index][2])

        return x1, x2, y1


def pil_loader(path, n_channels):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if n_channels == 1:
            return img.convert('L')
        elif n_channels == 3:
            return img.convert('RGB')
        else:
            raise ValueError('Invalid value for parameter n_channels!')
