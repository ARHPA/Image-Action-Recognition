from torchvision import datasets, transforms
from base import BaseDataLoader
from shutil import copyfile
from scipy.io import loadmat
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
            transforms.ToTensor(),
            transforms.Normalize((0.5077, 0.5077, 0.5077), (0.2120, 0.2120, 0.2120))
        ])
        self.data_dir = data_dir
        self.train_path = os.path.join(data_dir, "fer2013", 'train')
        self.train_set = datasets.ImageFolder(root=self.train_path, transform=trsfm)
        super().__init__(self.train_set, batch_size, shuffle, validation_split, num_workers)


class Stanford40(BaseDataLoader):
    """
    FER-2013 data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        creating_dataset()
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
            ),
            transforms.RandomGrayscale(0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.train_set = datasets.ImageFolder(root='StanfordActionDataset/train/',
                                              transform=transform_train)
        super().__init__(self.train_set, batch_size, shuffle, validation_split, num_workers)


def creating_dataset():
    images_path = "./JPEGImages"
    labels_path = "./ImageSplits"
    new_dataset_path = "./StanfordActionDataset"

    if not (os.path.exists(new_dataset_path)):
        os.mkdir(new_dataset_path)
        os.mkdir(new_dataset_path + '/' + 'train')
        os.mkdir(new_dataset_path + '/' + 'test')

    txts = os.listdir(labels_path)
    for txt in txts:
        idx = txt[0:-4].rfind('_')
        class_name = txt[0:idx]
        if class_name in ['actions.tx', 'test.tx', 'train.tx']:
            continue
        train_or_test = txt[idx + 1:-4]
        txt_contents = open(labels_path + '/' + txt)
        txt_contents = txt_contents.read()
        image_names = txt_contents.split('\n')
        num_aid_images_per_class = 1
        for image_name in image_names[0:-1]:
            if not (os.path.exists(new_dataset_path + '/' + train_or_test + '/' + class_name)):
                os.mkdir(new_dataset_path + '/' + train_or_test + '/' + class_name)
            copyfile(images_path + '/' + image_name,
                     new_dataset_path + '/' + train_or_test + '/' + class_name + '/' + image_name)
