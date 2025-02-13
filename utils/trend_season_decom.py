import torch
import torch.nn as nn
import numpy as np
import random

class ts_decom(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size=5, block_size=2, enhance_type='xsl', beta=0.01, gama=0.5):
        super(ts_decom, self).__init__()

        self.kernel_size = kernel_size
        self.block_size = block_size
        self.enhance_type = enhance_type
        self.beta = beta
        self.gama = gama

        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def moving_avg(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

    def sup_data(self, x, y):
        moving_mean = self.moving_avg(x)
        long_term_dep = x - moving_mean

        short_term_x = 0.05 * moving_mean
        long_term_x = 0.6 * long_term_dep
        x += short_term_x + long_term_x

        return x.cpu().numpy(), y.cpu().numpy()

    def count_class_num(self, train_target, num_classes):
        class_vector = np.zeros(num_classes)
        for i in range(num_classes):
            class_vector[i] = np.sum(train_target == i)

        class_inds_2d = [[] for _ in range(num_classes)]
        for i in range(len(train_target)):
            class_inds_2d[int(train_target[i])].append(i)

        return class_vector, class_inds_2d

    def common_pad(self, train_dataset, train_target, num_classes):
        class_vector, class_inds_2d = self.count_class_num(train_target, num_classes)
        max_num = max(class_vector)

        sup_vector = abs(class_vector - max_num)
        for i in range(len(sup_vector)):
            sv = int(sup_vector[i])
            if sv > 0:
                sup_inds = random.sample(class_inds_2d[i], min(sv, len(class_inds_2d[i])))

                sup_train_dataset, sup_train_target = train_dataset[sup_inds], train_target[sup_inds]
                sup_train_dataset, sup_train_target = (
                    self.sup_data(torch.from_numpy(sup_train_dataset).type(torch.FloatTensor).cuda(),
                                  torch.from_numpy(sup_train_target).type(torch.FloatTensor).cuda()))

                train_dataset = np.concatenate((train_dataset, sup_train_dataset), axis=0)
                train_target = np.concatenate((train_target, sup_train_target), axis=0)

        return train_dataset, train_target

    def forward(self, x, y):
        moving_mean = self.moving_avg(x)
        long_term_dep = x - moving_mean
        short_term_x = x + self.beta * moving_mean
        long_term_x = x + self.gama * long_term_dep

        res_x = torch.cat([x, short_term_x, long_term_x], dim=0)
        res_y = torch.cat([y, y, y], dim=0)

        return res_x.cpu().numpy(), res_y.cpu().numpy()
