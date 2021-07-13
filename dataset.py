from torch.utils.data import Dataset
from torchvision import transforms
from imageio import imread
import os


class MFI_WHU(Dataset):
    def __init__(self, path):
        super(MFI_WHU, self).__init__()
        self.path = path
        self.file_list = os.listdir(self.path + '/source_1')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(),
        ])

    def __getitem__(self, item):
        focus_1 = imread(self.path + '/source_1/' + self.file_list[item])
        focus_2 = imread(self.path + '/source_2/' + self.file_list[item])
        full_focus = imread(self.path + '/full_clear/' + self.file_list[item])
        return self.transform(focus_1), self.transform(focus_2), self.transform(full_focus)

    def __len__(self):
        return len(self.file_list)


class Lytro(Dataset):
    def __init__(self, path):
        super(Lytro, self).__init__()
        self.path = path
        self.file_list = os.listdir(self.path + '/Du Series/A')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(),
        ])

    def __getitem__(self, item):
        focus_1 = imread(self.path + '/Du Series/A/' + self.file_list[item])
        focus_2 = imread(self.path + '/Du Series/B/' + self.file_list[item].replace('A', 'B', 1))
        return self.transform(focus_1), self.transform(focus_2)

    def __len__(self):
        return len(self.file_list)