import torch
import torch.nn.utils.rnn as rnn_utils
from torchmetrics import F1Score, Accuracy, AveragePrecision, AUROC
from torchmetrics import MeanAbsoluteError, MeanSquaredLogError, MeanSquaredError, PearsonCorrCoef


def train(args, epoch, model, train_loader, valid_loader, device, criterion, optimizer):
    model.train()
    train_loss = 0
    metric_macro_acc = Accuracy(num_classes=args.classes, task='multilabel', num_labels=args.classes, average='macro').to(device)
    metric_macro_f1 = F1Score(average='macro', task='multilabel', num_labels=args.classes, num_classes=args.classes).to(device)
    metric_macro_ap = AveragePrecision(num_classes=args.classes, num_labels=args.classes, task='multilabel', threshold=.0).to(
        device)
    metric_auc = AUROC(num_classes=args.classes, task='multilabel', num_labels=args.classes, threshold=.0).to(device)

    for data in train_loader:
        voxel, seq, second_struct, gt, seq_lengths = data
        # print(seq_lengths)
        out = model(voxel.to(device))
        # out = model((voxel.to(device), seq.to(device), second_struct.to(device)), seq_lengths)
        # print(out[0])
        # print(gt[0])
        loss = criterion(out, gt.to(device).float())
        # loss_0 = criterion(out[0], gt.to(device).float())
        # loss_1 = criterion(out[1], gt.to(device).float())
        # loss_2 = criterion(out[2], gt.to(device).float())
        # loss = loss_0 + loss_1 + loss_2
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()

    model.eval()
    preds = []
    gt_list_valid = []
    with torch.no_grad():
        for data in valid_loader:
            voxel, seq, second_struct, gt, seq_lengths = data
            gt_list_valid.append(gt.cuda())
            out = model(voxel.to(device))

            # out = model((voxel.to(device), seq.to(device), second_struct.to(device)), seq_lengths)
            preds.append(out)

    # calculate metrics
    preds = torch.cat(preds, dim=0)
    gt_list_valid = torch.cat(gt_list_valid).int().squeeze(1)
    # class_ap = [round(i.item(), 5) for i in metric_class_ap(preds, gt_list_valid)]

    macro_ap = metric_macro_ap(preds, gt_list_valid).item()
    macro_auc = metric_auc(preds, gt_list_valid).item()
    macro_f1 = metric_macro_f1(preds, gt_list_valid).item()
    macro_acc = metric_macro_acc(preds, gt_list_valid).item()
    return train_loss, macro_ap, macro_f1, macro_acc, macro_auc


def train_reg(args, epoch, model, train_loader, valid_loader, device, criterion, optimizer):
    model.train()
    train_loss = 0
    metric_mae = MeanAbsoluteError().to(device)
    metric_mse = MeanSquaredError().to(device)
    metric_mlse = MeanSquaredLogError().to(device)
    metric_pcc = PearsonCorrCoef().to(device)
    for data in train_loader:
        voxel, seq, gt, seq_lengths = data
        # print(seq_lengths)
        out = model((voxel.to(device), seq.to(device)), seq_lengths)
        loss = criterion(out, gt.to(device).float())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()

    model.eval()
    preds = []
    gt_list_valid = []
    with torch.no_grad():
        for data in valid_loader:
            voxel, seq, gt, seq_lengths = data
            gt_list_valid.append(gt.cuda())
            out = model((voxel.to(device), seq.to(device)), seq_lengths)
            preds.append(out)

    # calculate metrics
    preds = torch.cat(preds, dim=0)
    gt_list_valid = torch.cat(gt_list_valid).int()

    mae = metric_mae(preds, gt_list_valid).item()
    mse = metric_mse(preds, gt_list_valid).item()
    mlse = metric_mlse(preds, gt_list_valid).item()
    # print(preds.shape)
    # print(gt_list_valid.shape)
    pcc = metric_pcc(preds.squeeze(-1), gt_list_valid.float().squeeze(-1)).item()
    return train_loss, mae, mse, mlse, pcc
