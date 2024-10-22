import argparse
import json
import logging
import os

import torch
from torch.utils.data import DataLoader

from dataset import ADataset
from model import SEQVAE, MMVAE, SEQWAE, MMWAE
from train import train


def main(args):
    weight_dir = os.path.join("./runs", args.gen_model+ "/" + args.model + str(args.seed))

    print('saving_dir: ', weight_dir)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    if not os.path.exists(weight_dir+'/weights/'):
        os.makedirs(weight_dir+'/weights/')

    logging.basicConfig(handlers=[
        logging.FileHandler(filename=os.path.join(weight_dir, "training.log"), encoding='utf-8', mode='w+')],
        format="%(asctime)s %(levelname)s:%(message)s", datefmt="%F %A %T", level=logging.INFO)

    with open(os.path.join(weight_dir, "config.json"), "w") as f:
        f.write(json.dumps(vars(args)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    f = open(weight_dir+'/loss_file.txt', 'w')
    if args.gen_model == 'vae':
        if args.model == 'seq':
            model = SEQVAE(args.classes, sequence_length=args.seq_length).cuda()
        elif args.model == 'mm_unet':
            model = MMVAE(args.classes, sequence_length=args.seq_length, use_encoder_feature=True).cuda()
        elif args.model == 'mm_mt':
            model = MMVAE(args.classes, sequence_length=args.seq_length, use_encoder_feature=False).cuda()
    elif args.gen_model == 'wae':
        if args.model == 'seq':
            model = SEQWAE(args.classes, sequence_length=args.seq_length).cuda()
        elif args.model == 'mm_unet':
            model = MMWAE(args.classes, sequence_length=args.seq_length, use_encoder_feature=True).cuda()
        elif args.model == 'mm_mt':
            model = MMWAE(args.classes, sequence_length=args.seq_length, use_encoder_feature=False).cuda()

    model.to(device)
    train_set = ADataset(mode='train', sequence_length=args.seq_length)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=True, weight_decay=5e-5)

    best_train_loss = 10000
    for epoch in range(1, args.epochs + 1):
        loss_sum, detailed_loss = train(args, epoch, model, train_loader, device, optimizer)

        line = ''
        for i in detailed_loss:
            line += (str(round(i, 3))+',')
        f.write('\n')
        torch.save(model.state_dict(), weight_dir + '/weights/' + str(epoch) + '.pth')
        if loss_sum < best_train_loss:
            best_train_loss = loss_sum
            torch.save(model.state_dict(), weight_dir+'/weights/best.pth')
            logging.info(f'Best Epoch: {epoch:03d},' + line)

    logging.info(f'Cross Validation Finished!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GOOD')
    parser.add_argument('--gen_model', type=str, default='vae',
                        help='vae wae ')
    parser.add_argument('--model', type=str, default='mm_unet',
                        help='model resnet26, bi-gru')
    parser.add_argument('--classes', type=str, default=4,
                        help='model')
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=64,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=80,
                        help='number of epochs to train (default: 80)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.002)')
    parser.add_argument('--decay', type=float, default=0.0005,
                        help='weight decay (default: 0.0005)')
    parser.add_argument('--warm-steps', type=int, dest='warm_steps', default=0,
                        help='number of warm start steps for learning rate (default: 10)')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience for early stopping (default: 10)')
    parser.add_argument('--loss', type=str, default='ce',
                        help='loss function (mlce, sl, mix)')

    parser.add_argument('--seed', type=int, default=1,
                        help="Seed for splitting dataset (default 1)")
    parser.add_argument('--pretrain', type=str, dest='pretrain', default='',
                        help='path of the pretrain model')

    parser.add_argument('--seq_length', type=int, default=30,
                        help="max length of peptides")
    parser.add_argument('--sup', type=bool, default=True,
                        help="whether use supervision")
    args = parser.parse_args()

    main(args)

