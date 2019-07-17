import time
import cv2

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from sklearn.model_selection import train_test_split

from efficientnet_pytorch import EfficientNet


class BlindnessDataset(Dataset):
    def __init__(self, df, path, size, transform=None):
        self.df = df
        self.path = path
        self.transform = transform
        self.size = size
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df.diagnosis.values[idx]
        code = self.df.id_code.values[idx]

        image = cv2.imread('' + self.path + '' + code + '.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = crop_image_from_gray(image)
        image = cv2.resize(image, (self.size, self.size))
        image = transforms.ToPILImage()(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class TrainEfficientNet():
    def __init__(self, **args):
        self.train_ds, self.val_ds= None, None
        self.model = None
        self.args = args

    def parse_holdout(self, df, path, rand_pct=0.2, seed=None):
        train_code, val_code = train_test_split(df, test_size=rand_pct, random_state=seed)

        self.train_ds = BlindnessDataset(train_code, path, 224)
        self.val_ds = BlindnessDataset(val_code, path, 224)

        return self.train_ds, self.val_ds

    def _check_train(self):
        if self.train_ds is None:
            raise Exception('training dataset is empty')

    def train(self, arch, valtype='holdout', gpu=1, epochs=10, lr=0.1):
        self._check_train()

        self.model = EfficientNet.from_pretrained(arch)
        
        criterion = nn.CrossEntropyLoss().cuda(gpu)
        optimizer = torch.optim.SGD(self.model.parameters(), lr)
        # optimizer = torch.optim.SGD(model.parameters(), lr,
        #                         momentum=momentum,weight_decay=weight_decay)

        start_epoch = 0
        for epoch in range(start_epoch, epochs):
            self.training(self.model, optimizer, criterion)
        #     validating()
        #     save_checkpoint()

    def training(self, model, optimizer, loss):
        model.train()

        train_loader = torch.utils.data.DataLoader(
            self.train_ds, batch_size=64, shuffle=False)

        for i, (images,targets) in enumerate(train_loader):
            # move to gpu

            output = model(images)
            loss = criterion(output, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def eval(self):
        pass

  