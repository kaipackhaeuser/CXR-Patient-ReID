from torch.utils import data
import numpy as np
from PIL import Image


class SiameseDatasetCovid19(data.Dataset):
    def __init__(self, phase='testing', n_channels=3, n_samples=792294, transform=None, image_path='./',
                 save_path=None):

        self.phase = phase
        self.n_samples = n_samples
        self.n_channels = n_channels
        self.transform = transform
        self.PATH = image_path

        if self.phase == 'testing':
            self.image_pairs = np.loadtxt('./image_pairs/pairs_testing_covid19.txt', dtype=str)

        if save_path is not None:
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
