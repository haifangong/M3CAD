import argparse
import json
import logging
import os
import time

from dataset import ADataset, HDataset, collate_fn
from swinunetr import SwinUNETR
from network import MMPeptide, SEQPeptide, VoxPeptide, MMFPeptide
from train import train, train_reg
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from loss import MLCE, SuperLoss
from utils import load_pretrain_model, set_seed


def main():
    parser = argparse.ArgumentParser(description='resnet26')
    # model setting
    parser.add_argument('--model', type=str, default='voxel-tr',
                        help='model resnet26, bi-gru')
    parser.add_argument('--fusion', type=str, default='1',
                        help="Seed for splitting dataset (default 1)")

    # task & dataset setting
    parser.add_argument('--task', type=str, default='anti',
                        help='task: anti toxin anti-all mechanism anti-binary anti-regression mic')
    parser.add_argument('--classes', type=int, default=6,
                        help='model')
    parser.add_argument('--split', type=int, default=5,
                        help="Split k fold in cross validation (default: 5)")
    parser.add_argument('--seed', type=int, default=1,
                        help="Seed for splitting dataset (default 1)")
    parser.add_argument('--gnn-type', type=str, dest='gnn_type', default="gat",
                        help='gnn type (gin, gcn, gat, graphsage)')

    # training setting
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=16,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='learning rate (default: 0.002)')
    parser.add_argument('--decay', type=float, default=0.0005,
                        help='weight decay (default: 0.0005)')
    parser.add_argument('--warm-steps', type=int, dest='warm_steps', default=0,
                        help='number of warm start steps for learning rate (default: 10)')
    parser.add_argument('--patience', type=int, default=25,
                        help='patience for early stopping (default: 10)')
    parser.add_argument('--loss', type=str, default='mlce',
                        help='loss function (mlce, sl, mix)')
    parser.add_argument('--pretrain', type=str, dest='pretrain', default='',
                        help='path of the pretrain model')  # /home/duadua/Desktop/fetal/3dpretrain/runs/e50.pth

    args = parser.parse_args()

    set_seed(args.seed)

    if args.loss == "mlce":
        criterion = MLCE()
    elif args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "smoothl1":
        criterion = nn.SmoothL1Loss()
    elif args.loss == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "bce":
        criterion = nn.BCELoss()
    else:
        raise NotImplementedError

    weight_dir = "./run/" + args.task + "-" + args.model + '-' + args.loss + str(args.batch_size) + str(args.lr) + str(args.epochs)
    print('saving_dir: ', weight_dir)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    
    logging.basicConfig(handlers=[
        logging.FileHandler(filename=os.path.join(weight_dir, "training.log"), encoding='utf-8', mode='w+')],
        format="%(asctime)s %(levelname)s:%(message)s", datefmt="%F %A %T", level=logging.INFO)

    with open(os.path.join(weight_dir, "config.json"), "w") as f:
        f.write(json.dumps(vars(args)))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info('Loading Test Dataset')

    best_perform_list = [[] for i in range(5)]

    for i in range(5):
        train_set = ADataset(mode='train', fold=i, task=args.task)
        test_set = ADataset(mode='valid', fold=i, task=args.task)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(test_set, batch_size=1, shuffle=False)
        if args.model == 'seq':
            model = SEQPeptide(classes=train_set.num_classes, q_encoder='mlp')
            args.classes = train_set.num_classes
        elif args.model == 'voxel-tr':
            model = SwinUNETR(img_size=(64, 64, 64), in_channels=3, classes=train_set.num_classes)
            args.classes = train_set.num_classes
        elif args.model == 'voxel':
            model = VoxPeptide(classes=train_set.num_classes)
            args.classes = train_set.num_classes
        elif args.model == 'mm':
            model = MMPeptide(classes=train_set.num_classes, q_encoder='mlp', ) # attention='hamburger'
            # model = MMPeptide(classes=train_set.num_classes, q_encoder='mlp', ) # attention='hamburger'
            args.classes = train_set.num_classes
        elif args.model == 'mmf':
            model = MMFPeptide(classes=train_set.num_classes, q_encoder='mlp', ) # attention='hamburger'
            args.classes = train_set.num_classes
        if len(args.pretrain) != 0:
            print('loading pretrain model')
            # model = load_pretrain_model(model, torch.load(args.pretrain))
            model_state = model.state_dict()
            pretrained_state = torch.load(args.pretrain)
            pretrained_state = {k: v for k, v in pretrained_state.items() if
                                k in model_state and v.size() == model_state[k].size()}
            model_state.update(pretrained_state)
            model.load_state_dict(model_state)
            # model.load_state_dict(torch.load(args.pretrain), strict=False)
        model.to(device)
        # optimizer = torch.optim.Adam(model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=True, weight_decay=5e-5)
        print(weight_dir)
        weights_path = f"{weight_dir}/model_{i + 1}.pth"
        # early_stopping = EarlyStopping(patience=args.patience, path=weights_path)
        logging.info(f'Running Cross Validation {i + 1}')

        best_metric = 0
        start_time = time.time()
        for epoch in range(1, args.epochs + 1):
            if train_set.num_classes == 1:
                train_loss, mae, mse, mlse, pcc = train_reg(args, epoch, model, train_loader, valid_loader, device, criterion, optimizer)
                print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, mae: {mae:.3f}, mse: {mse:.3f}, mlse: {mlse:.3f}, pcc: {pcc:.3f}')

                if mae > best_metric:
                    logging.info(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, mae: {mae:.3f}, mse: {mse:.3f}, mlse: {mlse:.3f}, pcc: {pcc:.3f}')
                    best_metric = mae
                    best_perform_list[i] = np.asarray([mae, mse, mlse, pcc])
                    torch.save(model.state_dict(), weights_path)

            else:
                train_loss, macro_ap, macro_f1, macro_acc, macro_auc = train(args, epoch, model, train_loader, valid_loader, device, criterion, optimizer)
                print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, macro_ap: {macro_ap:.3f}, macro_f1: {macro_f1:.3f}, macro_acc: {macro_acc:.3f}, macro_auc: {macro_auc:.3f}')
                avg_metric = macro_ap + macro_f1 + macro_acc + macro_auc
                if avg_metric > best_metric:
                    logging.info(
                        f'Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, macro_ap: {macro_ap:.3f}, macro_f1: {macro_f1:.3f}, macro_acc: {macro_acc:.3f}, macro_auc: {macro_auc:.3f}')
                    best_metric = avg_metric
                    best_perform_list[i] = np.asarray([macro_ap, macro_f1, macro_acc, macro_auc])
                    torch.save(model.state_dict(), weights_path)
            print('used time', time.time()-start_time)

    logging.info(f'Cross Validation Finished!')
    best_perform_list = np.asarray(best_perform_list)
    perform = open(weight_dir+'/result.txt', 'w')
    print(best_perform_list)
    print(np.mean(best_perform_list, 0))
    print(np.std(best_perform_list, 0))
    perform.write(','.join([str(i) for i in np.mean(best_perform_list, 0)])+'\n')
    perform.write(','.join([str(i) for i in np.std(best_perform_list, 0)])+'\n')


if __name__ == "__main__":
    main()
