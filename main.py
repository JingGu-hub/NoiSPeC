import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse

import warnings

from torch.utils.data import DataLoader

from models.model import Encoder, Classifier, Decoder
from utils.data_utils import reload_train_loader, TimeDataset
from utils.loss import Discrimn_Loss, delay_loss
from utils.trend_season_decom import ts_decom
from utils.utils import set_seed, build_dataset_pt, build_dataset_uea, flip_label, new_length, downsample_torch, \
    get_accuracy, create_dir, create_file, get_class_weight, refurb_label, count_refurb_matrix

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

# Base setup
parser.add_argument('--random_seed', type=int, default=42, help='shuffle seed')
parser.add_argument('--data_dir', type=str, default='./data/', help='dataset directory')
parser.add_argument('--model_save_dir', type=str, default='./outputs/model_save/', help='model save directory')
parser.add_argument('--result_save_dir', type=str, default='./outputs/result_save/', help='result save directory')
parser.add_argument('--model_names', type=list, default=['encoder1.pth', 'encoder2.pth', 'encoder3.pth'], help='model names')

# Dataset setup
parser.add_argument('--archive', type=str, default='UEA', help='Four, UEA')
parser.add_argument('--dataset', type=str, default='ArticularyWordRecognition', help='dataset name')  # [all dataset in Multivariate_arff]

# Label noise
parser.add_argument('--noise_flag', type=float, default=True, help='noise flag, True:make noisy label')
parser.add_argument('--label_noise_type', type=int, default=0, help='0 is Sym, 1 is Asym, -1 is Instance')
parser.add_argument('--noise_type', type=str, default='sym', help='0 is Sym, 1 is Asym, -1 is Instance')
parser.add_argument('--label_noise_rate', type=float, default=0.5, help='label noise ratio, sym: 0.2, 0.5, asym: 0.4, ins: 0.4')
parser.add_argument('--scale_list', type=list, default=[1, 2, 4], help='')

# training setup
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='')
parser.add_argument('--epoch', type=int, default=50, help='training epoch')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers')

# model setup
parser.add_argument('--embedding_size', type=int, default=16, help='model hyperparameters')
parser.add_argument('--feature_size', type=int, default=64, help='model output dimension')
parser.add_argument('--num_layers', type=int, default=2, help='number of  layers')
parser.add_argument('--num_heads', type=int, default=4, help='')

# feature augmentation and MCR setup
parser.add_argument('--alpha', type=float, default=0.01, help='MCR hyperparameters')
parser.add_argument('--beta', type=float, default=0.01, help='feature augmentation hyperparameters')
parser.add_argument('--gamma', type=float, default=0.5, help='feature augmentation hyperparameters')

# delay loss and refurb setup
parser.add_argument('--start_mask_epoch', type=int, default=10, help='sample mask epoch')
parser.add_argument('--start_delay_loss', type=int, default=15, help='start delayed loss epoch')
parser.add_argument('--delay_loss_k', type=int, default=3, help='the length of delay loss')
parser.add_argument('--start_refurb', type=int, default=30, help='start refurb epoch')
parser.add_argument('--refurb_len', type=int, default=5, help='the length of refurb')
parser.add_argument('--use_class_weight', type=float, default=False, help='')

# GPU setup
parser.add_argument('--gpu_id', type=int, default=0)

args = parser.parse_args()


def get_dataset(args, is_pretrain=False):
    if args.archive == 'Four':
        train_dataset, train_target, test_dataset, test_target, num_classes = build_dataset_pt(args)
    elif args.archive == 'UEA':
        train_dataset, train_target, test_dataset, test_target, num_classes = build_dataset_uea(args)

    # conduct feature enhance
    ts_aug = ts_decom(kernel_size=5, block_size=2, beta=0.01, gama=0.5)
    train_dataset, train_target = ts_aug(torch.from_numpy(train_dataset).type(torch.FloatTensor).cuda(),
                                         torch.from_numpy(train_target).type(torch.FloatTensor).cuda())

    # corrupt label
    if args.noise_flag == True:
        train_target, mask_train_target = flip_label(dataset=train_dataset, target=train_target, ratio=args.label_noise_rate,
                                                     args=args, pattern=args.label_noise_type)
    # supple data to mitigate class imbalance
    if is_pretrain:
        train_dataset, train_target = ts_aug.common_pad(train_dataset, train_target, num_classes)

    # load train_loader
    train_loader = reload_train_loader(args, train_dataset, train_target)
    # load test_loader
    test_set = TimeDataset(torch.from_numpy(test_dataset).type(torch.FloatTensor).cuda(),
                           torch.from_numpy(test_target).type(torch.FloatTensor).cuda().to(torch.int64))
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    return train_loader, test_loader, train_dataset, train_target, test_dataset, test_target, num_classes


def evaluate(args, epoch, test_loader, encoders, classifiers, last_five_accs, last_five_losses):
    encoder1, encoder2, encoder3 = encoders[0], encoders[1], encoders[2]
    classifier1, classifier2, classifier3 = classifiers[0], classifiers[1], classifiers[2]

    encoder1.eval()
    encoder2.eval()
    encoder3.eval()
    classifier1.eval()
    classifier2.eval()
    classifier3.eval()

    test_correct = 0
    epoch_test_loss = 0
    for x, y in test_loader:
        with torch.no_grad():
            input1 = downsample_torch(x, sample_rate=args.scale_list[0]).transpose(1, 2)
            input2 = downsample_torch(x, sample_rate=args.scale_list[1]).transpose(1, 2)
            input3 = downsample_torch(x, sample_rate=args.scale_list[2]).transpose(1, 2)

            # foreward
            encoder_output1 = encoder1(input1)
            encoder_output2 = encoder2(input2)
            encoder_output3 = encoder3(input3)
            c_out1 = classifier1(torch.cat([encoder_output1, encoder_output2], dim=1))
            c_out2 = classifier2(torch.cat([encoder_output1, encoder_output3], dim=1))
            c_out3 = classifier3(torch.cat([encoder_output2, encoder_output3], dim=1))

            loss1 = F.cross_entropy(c_out1, y, reduction='mean')
            loss2 = F.cross_entropy(c_out2, y, reduction='mean')
            loss3 = F.cross_entropy(c_out3, y, reduction='mean')

            loss = (loss1 + loss2 + loss3)
            epoch_test_loss += loss.item() * int(x.shape[0])
            test_correct += get_accuracy(c_out1, c_out2, c_out3, y)

    epoch_test_loss = epoch_test_loss / len(test_loader.dataset)
    test_acc = test_correct / len(test_loader.dataset)

    # compute last five accs and losses
    if (epoch + 5) >= args.epoch:
        last_five_accs.append(test_acc)
        last_five_losses.append(epoch_test_loss)

    return epoch_test_loss, test_acc, last_five_accs, last_five_losses

def pretrain(args, encoders, decoder, model_path, pretrain_optimizer):
    train_loader, test_loader, train_dataset, train_target, test_dataset, test_target, num_classes = get_dataset(args, is_pretrain=True)

    # train encoder and decoder
    encoder1, encoder2, encoder3 = encoders[0], encoders[1], encoders[2]

    mse_criterion = nn.MSELoss().cuda()
    discrimn_criterion = Discrimn_Loss().cuda()

    best_loss = float('inf')
    for epoch in range(args.epoch):
        encoder1.train()
        encoder2.train()
        encoder3.train()
        decoder.train()

        epoch_train_loss = 0
        for x, y in train_loader:
            # downsample
            input1 = downsample_torch(x, sample_rate=args.scale_list[0]).transpose(1, 2)
            input2 = downsample_torch(x, sample_rate=args.scale_list[1]).transpose(1, 2)
            input3 = downsample_torch(x, sample_rate=args.scale_list[2]).transpose(1, 2)
            x = x.transpose(1, 2)

            encoder_output1 = encoder1(input1)
            encoder_output2 = encoder2(input2)
            encoder_output3 = encoder3(input3)

            decoder_input = torch.cat([encoder_output1, encoder_output2, encoder_output3], dim=1)
            decoder_output = decoder(decoder_input)
            loss1 = mse_criterion(decoder_output, x)

            # compute MCR loss
            loss2 = discrimn_criterion(encoder_output1)
            loss3 = discrimn_criterion(encoder_output2)
            loss4 = discrimn_criterion(encoder_output3)
            discrimn_loss_total = ((loss2 + loss3 + loss4) / x.shape[0]) / 3

            loss = loss1 - args.alpha * discrimn_loss_total

            pretrain_optimizer.zero_grad()
            loss.backward()
            pretrain_optimizer.step()

            epoch_train_loss += loss.item() * x.shape[0]

        epoch_train_loss = epoch_train_loss / len(train_loader.dataset)

        encoder1.eval()
        encoder2.eval()
        encoder3.eval()
        decoder.eval()

        epoch_test_loss = 0
        for x, y in test_loader:
            with torch.no_grad():
                # downsample
                input1 = downsample_torch(x, sample_rate=args.scale_list[0]).transpose(1, 2)
                input2 = downsample_torch(x, sample_rate=args.scale_list[1]).transpose(1, 2)
                input3 = downsample_torch(x, sample_rate=args.scale_list[2]).transpose(1, 2)
                x = x.transpose(1, 2)

                encoder_output1 = encoder1(input1)
                encoder_output2 = encoder2(input2)
                encoder_output3 = encoder3(input3)

                decoder_input = torch.cat([encoder_output1, encoder_output2, encoder_output3], dim=1)
                decoder_output = decoder(decoder_input)
                loss1 = mse_criterion(decoder_output, x)

                # compute MCR loss
                loss2 = discrimn_criterion(encoder_output1)
                loss3 = discrimn_criterion(encoder_output2)
                loss4 = discrimn_criterion(encoder_output3)
                discrimn_loss_total = ((loss2 + loss3 + loss4) / x.shape[0]) / 3

                loss = loss1 - args.alpha * discrimn_loss_total
                epoch_test_loss += loss.item() * x.shape[0]

        epoch_test_loss = epoch_test_loss / len(test_loader.dataset)
        print('Epoch:', epoch, 'train Loss:', epoch_train_loss, 'test Loss:', epoch_test_loss)

        # save best model
        if epoch_test_loss < best_loss:
            best_loss = epoch_test_loss
            torch.save(encoder1.state_dict(), model_path + args.model_names[0])
            torch.save(encoder2.state_dict(), model_path + args.model_names[1])
            torch.save(encoder3.state_dict(), model_path + args.model_names[2])


def train(epoch, train_loader, train_dataset, train_target, encoders, classifiers, optimizer, refurb_matrixs,
          k_train_losses, unselected_inds, class_weight):
    encoder1, encoder2, encoder3 = encoders[0], encoders[1], encoders[2]
    classifier1, classifier2, classifier3 = classifiers[0], classifiers[1], classifiers[2]

    refurb_matrix1, refurb_matrix2, refurb_matrix3 = refurb_matrixs[0], refurb_matrixs[1], refurb_matrixs[2]
    k_train_loss1, k_train_loss2, k_train_loss3 = k_train_losses[0], k_train_losses[1], k_train_losses[2]

    encoder1.eval()
    encoder2.eval()
    encoder3.eval()
    classifier1.train()
    classifier2.train()
    classifier3.train()

    remember_rate = 1 if epoch < args.start_mask_epoch else 1 - args.label_noise_rate
    epoch_train_loss, epoch_train_correct = 0, 0
    loss_all = np.zeros(len(train_dataset))

    for x, y in train_loader:
        inds, y = y.chunk(2, dim=1)
        inds = inds.squeeze(1).detach().cpu().numpy()
        y = y.squeeze(1)

        # downsample
        input1 = downsample_torch(x, sample_rate=args.scale_list[0]).transpose(1, 2)
        input2 = downsample_torch(x, sample_rate=args.scale_list[1]).transpose(1, 2)
        input3 = downsample_torch(x, sample_rate=args.scale_list[2]).transpose(1, 2)

        # foreward
        encoder_output1 = encoder1(input1)
        encoder_output2 = encoder2(input2)
        encoder_output3 = encoder3(input3)
        c_out1 = classifier1(torch.cat([encoder_output1, encoder_output2], dim=1))
        c_out2 = classifier2(torch.cat([encoder_output1, encoder_output3], dim=1))
        c_out3 = classifier3(torch.cat([encoder_output2, encoder_output3], dim=1))

        # compute refurb matrix
        refurb_matrix1, refurb_matrix2, refurb_matrix3 = (
            count_refurb_matrix(c_out1, c_out2, c_out3, [refurb_matrix1, refurb_matrix2, refurb_matrix3], args.refurb_len, inds, epoch))

        # compute delay loss
        loss1, loss2, loss3, k_train_loss1, k_train_loss2, k_train_loss3, loss_all = (
            delay_loss(args, [c_out1, c_out2, c_out3], y, [k_train_loss1, k_train_loss2, k_train_loss3], epoch, loss_all, inds, class_weight))

        optimizer.zero_grad()
        loss1.backward(retain_graph=True)
        loss2.backward(retain_graph=True)
        loss3.backward()
        optimizer.step()

        loss = (loss1 + loss3 + loss3) / 3
        epoch_train_loss += loss.item() * int(x.shape[0] * remember_rate)
        epoch_train_correct += get_accuracy(c_out1, c_out2, c_out3, y)

    epoch_train_loss = epoch_train_loss / int(len(train_dataset) * remember_rate)
    epoch_train_acc = epoch_train_correct / len(train_dataset)

    # obtain unselected inds
    if epoch == args.start_mask_epoch:
        ind_1_sorted = torch.argsort(torch.from_numpy(loss_all).cuda(), descending=True)
        for i in range(int(len(ind_1_sorted) * args.label_noise_rate)):
            unselected_inds.append(ind_1_sorted[i].cpu().numpy().item())

    # refurb label
    if epoch >= args.start_refurb:
        train_target, noise_inds = refurb_label(train_loader, train_target, [refurb_matrix1, refurb_matrix2, refurb_matrix3], unselected_inds)
        # reload train loader
        train_loader = reload_train_loader(args, train_dataset, train_target)

    return epoch_train_loss, epoch_train_acc, train_loader, train_target, [encoder1, encoder2, encoder3], [classifier1, classifier2, classifier3], [refurb_matrix1, refurb_matrix2, refurb_matrix3], [k_train_loss1, k_train_loss2, k_train_loss3], unselected_inds


def main():
    torch.cuda.set_device(args.gpu_id)
    set_seed(args)

    # get dataset
    train_loader, test_loader, train_dataset, train_target, test_dataset, test_target, num_classes = get_dataset(args)

    input_dimension = train_dataset.shape[1]  # input feature dimension
    seq_length = train_dataset.shape[2]  # input sequence length

    # load model
    encoder1 = Encoder(new_length(seq_length, args.scale_list[0]), input_dimension, args.embedding_size, args.feature_size, args.num_layers, args.num_heads).cuda()
    encoder2 = Encoder(new_length(seq_length, args.scale_list[1]), input_dimension, args.embedding_size, args.feature_size, args.num_layers, args.num_heads).cuda()
    encoder3 = Encoder(new_length(seq_length, args.scale_list[2]), input_dimension, args.embedding_size, args.feature_size, args.num_layers, args.num_heads).cuda()
    decoder = Decoder(args.feature_size * len(args.scale_list), seq_length, input_dimension).cuda()

    classifier1 = Classifier(args.feature_size * 2, num_classes).cuda()
    classifier2 = Classifier(args.feature_size * 2, num_classes).cuda()
    classifier3 = Classifier(args.feature_size * 2, num_classes).cuda()

    # define delay loss
    k_train_loss1 = np.zeros((len(train_target), args.delay_loss_k))
    k_train_loss2 = np.zeros((len(train_target), args.delay_loss_k))
    k_train_loss3 = np.zeros((len(train_target), args.delay_loss_k))

    # define refurb matrix
    refurb_matrix1, refurb_matrix2, refurb_matrix3 = (np.zeros((len(train_target), args.refurb_len)),
                                                      np.zeros((len(train_target), args.refurb_len)),
                                                      np.zeros((len(train_target), args.refurb_len)))

    # define optimizer
    optimizer = torch.optim.Adam([{'params': encoder1.parameters()}, {'params': encoder2.parameters()}, {'params': encoder3.parameters()},
                                 {'params': classifier1.parameters()}, {'params': classifier2.parameters()}, {'params': classifier3.parameters()},
                                 {'params': decoder.parameters()}], lr=args.lr)

    # define start epoch and best loss
    unselected_inds = []
    last_five_accs, last_five_losses = [], []

    # define class weight
    class_weight = get_class_weight(args, train_target, num_classes)

    # pretrain model if model parameters do not exist
    model_path = args.model_save_dir + args.archive + '/' + str(args.dataset) + '/'
    create_dir(model_path)
    if not os.path.exists(args.model_save_dir + args.archive + '/' + str(args.dataset) + '/' + args.model_names[0]):
        # if not exist, pretrain and save model
        pretrain(args, [encoder1, encoder2, encoder3], decoder, model_path, optimizer)

    # load pretrain model
    encoder1.load_state_dict(torch.load(model_path + args.model_names[0]))
    encoder2.load_state_dict(torch.load(model_path + args.model_names[1]))
    encoder3.load_state_dict(torch.load(model_path + args.model_names[2]))

    out_dir = args.result_save_dir + args.archive + '/' + str(args.dataset) + '/'
    out_file = out_dir + '%s_%s%.1f.txt' % (args.dataset, args.noise_type, args.label_noise_rate)
    out_file = create_file(out_dir, out_file, 'epoch,train loss,train acc,test loss,test acc\n')
    for epoch in range(args.epoch):
        # train
        train_loss, train_acc, train_loader, train_target, encoders, classifiers, refurb_matrixs, k_train_losses, unselected_inds = (
            train(epoch, train_loader, train_dataset, train_target, [encoder1, encoder2, encoder3],[classifier1, classifier2, classifier3], optimizer,
                  [refurb_matrix1, refurb_matrix2, refurb_matrix3], [k_train_loss1, k_train_loss2, k_train_loss3], unselected_inds, class_weight))

        encoder1, encoder2, encoder3 = encoders[0], encoders[1], encoders[2]
        classifier1, classifier2, classifier3 = classifiers[0], classifiers[1], classifiers[2]
        refurb_matrix1, refurb_matrix2, refurb_matrix3 = refurb_matrixs[0], refurb_matrixs[1], refurb_matrixs[2]
        k_train_loss1, k_train_loss2, k_train_loss3 = k_train_losses[0], k_train_losses[1], k_train_losses[2]

        # test
        test_loss, test_acc, last_five_accs, last_five_losses = evaluate(args, epoch, test_loader, encoders,
                                                                         classifiers, last_five_accs, last_five_losses)
        print('Epoch:[%d/%d] train_loss:%f, train_acc:%f, test_loss:%f, test_acc:%f' % (
        epoch + 1, args.epoch, train_loss, train_acc, test_loss, test_acc))

        with open(out_file, "a") as myfile:
            myfile.write(str('Epoch:[%d/%d] train_loss:%f, train_acc:%f, test_loss:%f, test_acc:%f' % (
            epoch + 1, args.epoch, train_loss, train_acc, test_loss, test_acc) + "\n"))

    test_accuracy = round(np.mean(last_five_accs), 4)
    test_loss = round(np.mean(last_five_losses), 4)
    print('Test Accuracy:', test_accuracy, 'Test Loss:', test_loss)


if __name__ == '__main__':
    main()
