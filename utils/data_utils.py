import torch
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader

class TimeDataset(data.Dataset):
    def __init__(self, dataset, target):
        self.dataset = dataset
        # self.dataset = np.expand_dims(self.dataset, 1)
        # print("dataset shape = ", dataset.shape)
        if len(self.dataset.shape) == 2:
            self.dataset = torch.unsqueeze(self.dataset, 1)
        # print("dataset shape = ", self.dataset.shape)
        self.target = target

    def __getitem__(self, index):
        return self.dataset[index], self.target[index]

    def __len__(self):
        return len(self.target)

def reload_train_loader(args, train_dataset, train_target):
    indices = torch.from_numpy(np.arange(len(train_target))).cuda().reshape(-1, 1)
    temp_train_target = torch.from_numpy(train_target).type(torch.FloatTensor).cuda().to(torch.int64).reshape(-1, 1)
    temp_train_target = torch.cat([indices, temp_train_target], dim=1)

    train_set = TimeDataset(torch.from_numpy(train_dataset).type(torch.FloatTensor).cuda(), temp_train_target)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    return train_loader