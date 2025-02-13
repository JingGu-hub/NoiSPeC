import os
import numpy as np
import pandas as pd
import random
import torch
import torch.utils.data as data
import torch.nn as nn
from scipy.io.arff import loadarff
from scipy import stats
import torch.nn.functional as F
from math import inf
from datetime import datetime

def set_seed(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

def build_dataset_uea(args):
    data_path = args.data_dir
    train_data = loadarff(os.path.join(data_path, args.dataset, args.dataset + '_TRAIN.arff'))[0]
    test_data = loadarff(os.path.join(data_path, args.dataset, args.dataset + '_TEST.arff'))[0]

    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([d.tolist() for d in t_data])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)

    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)

    train_dataset = train_X.transpose(0, 2, 1)
    train_target = train_y
    test_dataset = test_X.transpose(0, 2, 1)
    test_target = test_y

    num_classes = len(np.unique(train_target))
    train_target = transfer_labels(train_target)
    test_target = transfer_labels(test_target)

    ind = np.where(np.isnan(train_dataset))
    col_mean = np.nanmean(train_dataset, axis=0)
    col_mean[np.isnan(col_mean)] = 1e-6

    train_dataset[ind] = np.take(col_mean, ind[1])

    ind_test = np.where(np.isnan(test_dataset))
    test_dataset[ind_test] = np.take(col_mean, ind_test[1])

    train_dataset, train_target = shuffler_dataset(train_dataset, train_target)
    test_dataset, test_target = shuffler_dataset(test_dataset, test_target)

    return train_dataset, train_target, test_dataset, test_target, num_classes

def build_dataset_pt(args):
    data_path = args.data_dir + args.dataset
    train_dataset_dict = torch.load(os.path.join(data_path, "train.pt"))
    train_dataset = train_dataset_dict["samples"].numpy()  # (num_size, num_dimensions, series_length)
    train_target = train_dataset_dict["labels"].numpy()
    num_classes = len(np.unique(train_dataset_dict["labels"].numpy(), return_counts=True)[0])
    train_target = transfer_labels(train_target)

    test_dataset_dict = torch.load(os.path.join(data_path, "test.pt"))
    test_dataset = test_dataset_dict["samples"].numpy()  # (num_size, num_dimensions, series_length)
    test_target = test_dataset_dict["labels"].numpy()
    test_target = transfer_labels(test_target)

    return train_dataset, train_target, test_dataset, test_target, num_classes

def transfer_labels(labels):
    indicies = np.unique(labels)
    num_samples = labels.shape[0]

    for i in range(num_samples):
        new_label = np.argwhere(labels[i] == indicies)[0][0]
        labels[i] = new_label

    return labels


def shuffler_dataset(x_train, y_train):
    indexes = np.array(list(range(x_train.shape[0])))
    np.random.shuffle(indexes)
    x_train = x_train[indexes]
    y_train = y_train[indexes]
    return x_train, y_train


def flip_label(dataset, target, ratio, args=None, pattern=0):
    """
    Induce label noise by randomly corrupting labels
    :param target: list or array of labels
    :param ratio: float: noise ratio
    :param pattern: flag to choose which type of noise.
            0 or mod(pattern, #classes) == 0 = symmetric
            int = asymmetric
            -1 = flip
    :return:
    """
    assert 0 <= ratio < 1

    target = np.array(target).astype(int)
    label = target.copy()
    n_class = len(np.unique(label))

    if type(pattern) is int:
        if pattern == -1:
            # Instance
            num_classes = len(np.unique(target, return_counts=True)[0])
            data = torch.from_numpy(dataset).type(torch.FloatTensor)
            targets = torch.from_numpy(target).type(torch.FloatTensor).to(torch.int64)
            dataset_ = zip(data, targets)
            feature_size = dataset.shape[1] * dataset.shape[2]
            label = get_instance_noisy_label(n=ratio, dataset=dataset_, labels=targets, num_classes=num_classes,
                                             feature_size=feature_size, seed=args.random_seed if args is not None else 42)
        else:
            for i in range(label.shape[0]):
                # symmetric noise
                if (pattern % n_class) == 0:
                    p1 = ratio / (n_class - 1) * np.ones(n_class)
                    p1[label[i]] = 1 - ratio
                    label[i] = np.random.choice(n_class, p=p1)
                elif pattern == 1:
                    # Asymm
                    label[i] = np.random.choice([label[i], (target[i] + pattern) % n_class], p=[1 - ratio, ratio])

    elif type(pattern) is str:
        raise ValueError

    mask = np.array([int(x != y) for (x, y) in zip(target, label)])

    return label, mask

def new_length(seq_length, sample_rate):
    last_one = 0
    if seq_length % sample_rate > 0:
        last_one = 1
    new_length = int(np.floor(seq_length / sample_rate)) + last_one
    return new_length

def downsample_torch(x_data, sample_rate):
    """
     Takes a batch of sequences with [batch size, channels, seq_len] and down-samples with sample
     rate k. hence, every k-th element of the original time series is kept.
    """
    last_one = 0
    if x_data.shape[2] % sample_rate > 0:
        last_one = 1

    new_length = int(np.floor(x_data.shape[2] / sample_rate)) + last_one
    output = torch.zeros((x_data.shape[0], x_data.shape[1], new_length)).cuda()
    output[:, :, range(new_length)] = x_data[:, :, [i * sample_rate for i in range(new_length)]]

    return output

def get_instance_noisy_label(n, dataset, labels, num_classes, feature_size, norm_std=0.1, seed=42):
    # n -> noise_rate
    # dataset
    # labels -> labels (targets)
    # label_num -> class number
    # feature_size
    # norm_std -> default 0.1
    # seed -> random_seed

    label_num = num_classes
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    P = []
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=norm_std)
    flip_rate = flip_distribution.rvs(labels.shape[0])

    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)
    labels = labels.cuda()

    W = np.random.randn(label_num, feature_size, label_num)

    W = torch.FloatTensor(W).cuda()
    for i, (x, y) in enumerate(dataset):
        # 1*m *  m*10 = 1*10
        x = x.cuda()
        t = W[y]
        A = x.reshape(1, -1).mm(W[y]).squeeze(0)

        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy()

    l = [i for i in range(label_num)]
    new_label = [np.random.choice(l, p=P[i]) for i in range(labels.shape[0])]
    print(f'noise rate = {(new_label != np.array(labels.cpu())).mean()}')

    record = [[0 for _ in range(label_num)] for i in range(label_num)]

    for a, b in zip(labels, new_label):
        a, b = int(a), int(b)
        record[a][b] += 1
        #
    print('****************************************')
    print('following is flip percentage:')

    for i in range(label_num):
        sum_i = sum(record[i])
        for j in range(label_num):
            if i != j:
                print(f"{record[i][j] / sum_i: .2f}", end='\t')
            else:
                print(f"{record[i][j] / sum_i: .2f}", end='\t')
        print()

    pidx = np.random.choice(range(P.shape[0]), 1000)
    cnt = 0
    for i in range(1000):
        if labels[pidx[i]] == 0:
            a = P[pidx[i], :]
            for j in range(label_num):
                print(f"{a[j]:.2f}", end="\t")
            print()
            cnt += 1
        if cnt >= 10:
            break
    print(P)
    return np.array(new_label)


def get_clean_loss_tensor_mask(loss_all, remember_rate):
    '''
    :param loss: numpy
    :param remember_rate: float, 1 - noise_rate
    :return: mask_loss, 1 is clean, 0 is noise
    '''

    ind_1_sorted =  torch.argsort(loss_all)
    mask_loss = torch.zeros(len(ind_1_sorted)).cuda()
    for i in range(int(len(ind_1_sorted) * remember_rate)):
        mask_loss[ind_1_sorted[i]] = 1  ## 1 is samll loss (clean), 0 is big loss (noise)

    return mask_loss

def get_accuracy(classifier_output1, classifier_output2, classifier_output3, y):
    target_pred1 = torch.argmax(classifier_output1.data, axis=1)
    target_pred2 = torch.argmax(classifier_output2.data, axis=1)
    target_pred3 = torch.argmax(classifier_output3.data, axis=1)

    target_pred1 = target_pred1.unsqueeze(1)
    target_pred2 = target_pred2.unsqueeze(1)
    target_pred3 = target_pred3.unsqueeze(1)

    final_target_pred = torch.cat((target_pred1, target_pred2, target_pred3), 1)
    target_pred_temp = torch.mode(final_target_pred, axis=1).values

    return target_pred_temp.eq(y).sum().item()

def count_class_num(train_target, num_classes):
    class_vector = np.zeros(num_classes)
    for i in range(num_classes):
        class_vector[i] = np.sum(train_target == i)
    class_vector = [int(item) for item in class_vector]

    return class_vector

def get_class_weight(args, train_target, num_classes):
    if args.use_class_weight:
        class_vector = np.zeros(num_classes)
        avg_class_num = len(train_target) / num_classes
        for i in range(num_classes):
            class_vector[i] = np.sum(train_target == i) / avg_class_num

        class_weight = np.zeros(len(train_target))
        for i in range(len(train_target)):
            class_weight[i] = class_vector[int(train_target[i])]
        class_weight = torch.from_numpy(class_weight).float().cuda()
    else:
        class_weight = torch.ones(len(train_target)).float().cuda()

    return class_weight

def make_dir(base_path, paths=[], is_clinodiagonal=False):
    for path in paths:
        if not os.path.exists(os.path.join(base_path, path)):
            os.makedirs(os.path.join(base_path, path))
        base_path = os.path.join(base_path, path)

    return base_path if is_clinodiagonal==False else base_path + '/'

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def create_file(path, filename, write_line):
    create_dir(path)

    new_file_name = None
    if filename != None:
        nowTime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        if not os.path.exists(filename):
            print("create new file: %s" % filename)
        else:
            new_file_name = filename + ".bak-%s" % nowTime
            os.system('mv %s %s' % (filename, new_file_name))

        new_file_name = new_file_name if new_file_name != None else filename
        if write_line != None:
            with open(new_file_name, "a") as myfile:
                myfile.write(write_line + '\n')

    return new_file_name

def count_refurb_matrix(classifier_output1, classifier_output2, classifier_output3, refurb_matrixs, refurb_len, inds, epoch):
    refurb_matrix1, refurb_matrix2, refurb_matrix3 = refurb_matrixs[0], refurb_matrixs[1], refurb_matrixs[2]

    train_target_prob1, target_pred1 = classifier_output1.max(1)
    train_target_prob2, target_pred2 = classifier_output2.max(1)
    train_target_prob3, target_pred3 = classifier_output3.max(1)

    refurb_matrix1[inds, epoch % refurb_len] = target_pred1.cpu().numpy()
    refurb_matrix2[inds, epoch % refurb_len] = target_pred2.cpu().numpy()
    refurb_matrix3[inds, epoch % refurb_len] = target_pred3.cpu().numpy()

    return refurb_matrix1, refurb_matrix2, refurb_matrix3

def refurb_label(train_loader, train_target, refurb_matrixs, noise_inds):
    refurb_matrix1, refurb_matrix2, refurb_matrix3 = refurb_matrixs[0], refurb_matrixs[1], refurb_matrixs[2]

    train_target_pred_mode1 = torch.mode(torch.from_numpy(refurb_matrix1).cuda(), axis=1).values.unsqueeze(1)
    train_target_pred_mode2 = torch.mode(torch.from_numpy(refurb_matrix2).cuda(), axis=1).values.unsqueeze(1)
    train_target_pred_mode3 = torch.mode(torch.from_numpy(refurb_matrix3).cuda(), axis=1).values.unsqueeze(1)

    pred_label = torch.cat((train_target_pred_mode1, train_target_pred_mode2, train_target_pred_mode3), dim=1)
    refurb_y = torch.mode(pred_label, axis=1).values.cpu().numpy()

    for x, y in train_loader:
        inds, y = y.chunk(2, dim=1)
        inds = inds.squeeze(1).detach().cpu().numpy()

        for i in range(len(inds)):
            ry_n = np.max(np.bincount(pred_label[inds[i]].detach().cpu().numpy().astype(int)))
            if ry_n == pred_label.shape[1] and train_target[inds[i]] != refurb_y[inds[i]] and inds[i] in noise_inds:
                train_target[inds[i]] = refurb_y[inds[i]]
                noise_inds.remove(inds[i])

    return train_target, noise_inds
