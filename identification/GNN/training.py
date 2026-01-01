import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torchmetrics.functional import pearson_corrcoef
from torchmetrics import F1Score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args, epoch, model, train_loader, valid_loader, device, criterion, optimizer):
    model.train()
    total_train_loss = 0
    train_data_size = 0
    f1s = F1Score(average='macro', num_classes=3, threshold=0).to(device)
    f1s2f = F1Score(average='none', num_classes=3, threshold=0).to(device)
    bce_loss = nn.BCELoss()
    var_list = []
    abs_correct_rate = [0, 0, 0]
    re_correct_rate = [0, 0, 0]
    curri1 = []
    curri = []
    encodings, labels = [], []
    criterion0, criterion1 = criterion
    yt = []
    for data in train_loader:
        data = data.to(device)
        out = model(data)
        loss = criterion0(out, torch.tensor(np.asarray(data.y)).cuda().float()) 
        # pred = torch.cat((out), dim=1)
        # loss = bce_loss(torch.sigmoid(pred), torch.tensor(np.asarray(data.y)).cuda().float())
        yt.append(torch.tensor(np.asarray(data.y)).cuda())

        # for idx in range(pred.shape[1]):
        #     loss += bce_loss(out[idx], torch.tensor(np.asarray(data.y[idx])).cuda().float())
            
        # + (1 - epoch/500) * criterion1(out, torch.tensor(np.asarray(data.y)).cuda().float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_train_loss += loss.item()
    if epoch==1:
        yt = torch.cat(yt).int()
        print(torch.sum(yt,0))
    train_loss = total_train_loss
    # print(var_list)
    del var_list
    model.eval()
    total_valid_loss = 0
    valid_data_size = 0
    preds = []
    ys = []
    with torch.no_grad():
        for data in valid_loader:
            data = data.to(device)
            out = model(data)
            pred = out
            # pred = torch.cat((out), dim=1)
            preds.append(pred)
            ys.append(torch.tensor(np.asarray(data.y)).cuda())
    preds = torch.cat(preds, dim=0)
    ys = torch.cat(ys).int()
    if epoch==1:
        print(torch.sum(ys,0))
    # valid_loss = total_valid_loss / valid_data_size
    # print('preds', preds.shape)
    # print('ys', ys.shape)
    # print(preds.shape)
    # print(ys.shape)
    valid_loss = f1s(preds, ys)
    class_specific_f1 = f1s2f(preds, ys)
    return train_loss, valid_loss, class_specific_f1


def evaluate(args, model, loader, device, return_tensor=False):
    model.eval()
    auc_pred, auc_label = [], []
    pred = []
    y = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if args.fds or args.contrast_curri:
                out, _ = model(data)
            else:
                out = model(data)
            pred.append(out)
            y.append(data.y)
            auc_pred.extend(out.cpu().numpy().reshape(-1).tolist())
            auc_label.extend(data.y.cpu().numpy().reshape(-1).tolist())

        pred_tensor = torch.cat(pred)
        y_tensor = torch.cat(y)
        corr = pearson_corrcoef(pred_tensor, y_tensor)
        rmse = torch.sqrt(mse_loss(pred_tensor, y_tensor))

    if return_tensor:
        return pred_tensor, y_tensor
    else:
        return corr, rmse


def metrics(pred_dir, pred_rev, y_dir, y_rev):
    corr_dir = pearson_corrcoef(pred_dir, y_dir)
    rmse_dir = torch.sqrt(mse_loss(pred_dir, y_dir))
    corr_rev = pearson_corrcoef(pred_rev, y_rev)
    rmse_rev = torch.sqrt(mse_loss(pred_rev, y_rev))
    corr_dir_rev = pearson_corrcoef(pred_dir, pred_rev)
    delta = torch.mean(pred_dir + pred_rev)

    return corr_dir, rmse_dir, corr_rev, rmse_rev, corr_dir_rev, delta


class EarlyStopping:

    def __init__(self, patience=10, path='checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path

    def __call__(self, score, model, goal="maximize"):

        if goal == "minimize":
            score = -score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)

        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        torch.save(model.state_dict(), self.path)
        self.best_score = score
