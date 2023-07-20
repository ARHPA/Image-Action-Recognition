from torchvision import datasets, transforms
from base import BaseDataLoader
import os


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class FER(BaseDataLoader):
    """
    FER-2013 data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5077, 0.5077, 0.5077), (0.2120, 0.2120, 0.2120))
        ])
        self.data_dir = data_dir
        self.train_path = os.path.join(data_dir, "fer2013", 'train')
        self.train_set = datasets.ImageFolder(root=self.train_path, transform=trsfm)
        super().__init__(self.train_set, batch_size, shuffle, validation_split, num_workers)

