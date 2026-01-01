import argparse
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torchmetrics import Accuracy, F1Score, AveragePrecision, AUROC

from GNN.model import GraphGNN
from dataset_graph import load_dataset
from loss import MLCE
from utils import set_seed


def get_criterion(loss_type):
    if loss_type == "mlce":
        return MLCE()
    elif loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "smoothl1":
        return nn.SmoothL1Loss()
    elif loss_type == "ce":
        return nn.CrossEntropyLoss()
    elif loss_type == "bce":
        return nn.BCELoss()
    else:
        raise NotImplementedError(f"Loss {loss_type} not supported")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        # data_graph is the first element of the tuple returned by the loader
        data_graph = data[0].to(device)
        targets = torch.tensor(np.asarray(data_graph.gt)).to(device).float()
        
        optimizer.zero_grad()
        out = model(data_graph)
        
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, metrics, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    for data in loader:
        data_graph = data[0].to(device)
        targets = torch.tensor(np.asarray(data_graph.gt)).to(device).int()
        
        out = model(data_graph)
        preds = torch.sigmoid(out)
        
        all_preds.append(preds)
        all_targets.append(targets)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    results = {name: metric(all_preds, all_targets).item() for name, metric in metrics.items()}
    return results


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup directories and logging
    weight_dir = f"./run/{args.task}-{args.gnn_type}-{args.loss}{args.batch_size}{args.lr}{args.epochs}"
    os.makedirs(weight_dir, exist_ok=True)

    logging.basicConfig(
        handlers=[logging.FileHandler(os.path.join(weight_dir, "training.log"), mode='w+')],
        format="%(asctime)s %(levelname)s:%(message)s", 
        level=logging.INFO
    )
    with open(os.path.join(weight_dir, "config.json"), "w") as f:
        json.dump(vars(args), f)

    # Load Data
    logging.info('Loading Dataset')
    data_list = load_dataset(args.task)
    random.shuffle(data_list)

    # Metrics Setup
    metrics = {
        "acc": Accuracy(task='multilabel', num_labels=args.classes, average='macro').to(device),
        "f1": F1Score(task='multilabel', num_labels=args.classes, average='macro').to(device),
        "ap": AveragePrecision(task='multilabel', num_labels=args.classes, average='macro').to(device),
        "auc": AUROC(task='multilabel', num_labels=args.classes, average='macro').to(device)
    }

    # 5-Fold Cross Validation
    fold_results = []
    indices = list(range(len(data_list)))
    
    for fold in range(args.split):
        logging.info(f'Starting Fold {fold + 1}/{args.split}')
        
        valid_idx = indices[fold::args.split]
        train_idx = [i for i in indices if i not in valid_idx]
        
        train_set = [data_list[i] for i in train_idx]
        valid_set = [data_list[i] for i in valid_idx]

        train_loader = DataLoader(train_set, batch_size=args.batch_size, follow_batch=['x_s'], shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, follow_batch=['x_s'], shuffle=False)

        model = GraphGNN(
            num_layer=args.num_layer, input_dim=20, emb_dim=args.emb_dim, 
            out_dim=args.classes, JK="last", drop_ratio=args.dropout_ratio, 
            graph_pooling=args.graph_pooling, gnn_type=args.gnn_type
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)
        criterion = get_criterion(args.loss)
        
        best_fold_score = -1
        best_fold_metrics = None
        weights_path = os.path.join(weight_dir, f"model_{fold + 1}.pth")

        for epoch in tqdm(range(1, args.epochs + 1)):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            res = evaluate(model, valid_loader, metrics, device)
            
            # Simple heuristic for "best" model: sum of metrics
            current_score = res['ap'] + res['f1'] + res['acc'] + res['auc']
            if current_score > best_fold_score:
                best_fold_score = current_score
                best_fold_metrics = [res['ap'], res['f1'], res['acc'], res['auc']]
                torch.save(model.state_dict(), weights_path)
            
            if epoch % 10 == 0:
                logging.info(f"Fold {fold+1} Epoch {epoch}: Loss {train_loss:.4f} AP {res['ap']:.4f}")

        fold_results.append(best_fold_metrics)

    # Aggregate and Save Results
    fold_results = np.array(fold_results)
    mean_res = np.mean(fold_results, axis=0)
    std_res = np.std(fold_results, axis=0)

    print(f"Final Mean Results: {mean_res}")
    with open(os.path.join(weight_dir, 'result.txt'), 'w') as f:
        f.write(','.join(map(str, mean_res)) + '\n')
        f.write(','.join(map(str, std_res)) + '\n')
    logging.info("Cross Validation Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GNN Training')
    parser.add_argument('--task', type=str, default='anti')
    parser.add_argument('--classes', type=int, default=6)
    parser.add_argument('--split', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num-layer', type=int, default=1)
    parser.add_argument('--emb-dim', type=int, default=60)
    parser.add_argument('--dropout-ratio', type=float, default=0.0)
    parser.add_argument('--graph-pooling', type=str, default="attention")
    parser.add_argument('--gnn-type', type=str, default="graphsage")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--decay', type=float, default=0.0005)
    parser.add_argument('--loss', type=str, default='ce')
    
    args = parser.parse_args()
    main(args)
