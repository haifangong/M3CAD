import torch
from torchmetrics import F1Score, Accuracy, AveragePrecision, AUROC, KendallRankCorrCoef
from torchmetrics import MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef


def train(args, epoch, model, train_loader, valid_loader, device, criterion, optimizer):
    """
    Train the identification model for one epoch.

    Args:
        args: Command line arguments.
        epoch (int): Current epoch number.
        model: PyTorch model (MMPeptide).
        train_loader: DataLoader for training.
        valid_loader: DataLoader for validation.
        device: Torch device.
        criterion: Loss function (e.g., MSELoss, CrossEntropyLoss).
        optimizer: Optimizer.

    Returns:
        tuple: Training loss and various validation metrics (AP, F1, Accuracy, AUC).
    """
    model.train()
    train_loss = 0
    
    # Initialize metric calculators (Macro average)
    metric_macro_acc = Accuracy(num_classes=args.classes, task=args.metric, num_labels=args.classes, average='macro').to(device)
    metric_macro_f1 = F1Score(average='macro', task=args.metric, num_labels=args.classes, num_classes=args.classes).to(device)
    metric_macro_ap = AveragePrecision(num_classes=args.classes, num_labels=args.classes, task=args.metric).to(device) # , threshold=.0
    metric_auc = AUROC(num_classes=args.classes, task=args.metric, num_labels=args.classes).to(device) # , threshold=.0
    
    # Initialize per-class metric calculators
    metric_per_class_ap = AveragePrecision(num_classes=args.classes, num_labels=args.classes, task=args.metric, average=None).to(device)
    metric_per_class_auc = AUROC(num_classes=args.classes, task=args.metric, num_labels=args.classes, average=None).to(device)
    metric_per_class_f1 = F1Score(num_classes=args.classes, task=args.metric, num_labels=args.classes, average=None, threshold=0.5).to(device)
    metric_per_class_acc = Accuracy(num_classes=args.classes, task=args.metric, num_labels=args.classes, average=None, threshold=0.5).to(device)

    # --- Training Loop ---
    for data in train_loader:
        voxel, seq, second_struct, gt, seq_lengths = data
        
        # Forward pass
        out, feature = model((voxel.to(device), seq.to(device)))
        
        # Compute loss
        loss = criterion(out, gt.to(device).float())
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()

    # --- Validation Loop ---
    model.eval()
    preds = []
    gt_list_valid = []
    with torch.no_grad():
        for data in valid_loader:
            voxel, seq, second_struct, gt, seq_lengths = data
            gt_list_valid.append(gt.cuda())
            out, feature = model((voxel.to(device), seq.to(device)))
            preds.append(out)

    # --- Metrics Calculation ---
    preds = torch.cat(preds, dim=0)
    gt_list_valid = torch.cat(gt_list_valid).int().squeeze(1)

    # Calculate Macro metrics
    macro_ap = metric_macro_ap(preds.squeeze(1), gt_list_valid).item()
    macro_auc = metric_auc(preds.squeeze(1), gt_list_valid).item()
    macro_f1 = metric_macro_f1(preds.squeeze(1), gt_list_valid).item()
    macro_acc = metric_macro_acc(preds.squeeze(1), gt_list_valid).item()
    
    # Calculate Per-class metrics
    per_class_ap = metric_per_class_ap(preds.squeeze(1), gt_list_valid).cpu().numpy()
    per_class_auc = metric_per_class_auc(preds.squeeze(1), gt_list_valid).cpu().numpy()
    per_class_f1 = metric_per_class_f1(preds.squeeze(1), gt_list_valid).cpu().numpy()
    per_class_acc = metric_per_class_acc(preds.squeeze(1), gt_list_valid).cpu().numpy()
    
    return train_loss, macro_ap, macro_f1, macro_acc, macro_auc, per_class_ap, per_class_auc, per_class_f1, per_class_acc


def train_reg(args, epoch, model, train_loader, valid_loader, device, criterion, optimizer):
    """
    Train the regression model for one epoch.

    Args:
        args: Command line arguments.
        epoch (int): Current epoch number.
        model: PyTorch model (MMPeptide).
        train_loader: DataLoader for training.
        valid_loader: DataLoader for validation.
        device: Torch device.
        criterion: Loss function (e.g., MSELoss).
        optimizer: Optimizer.

    Returns:
        tuple: (Train Loss, MAE, MSE, PCC, KCC)
    """
    model.train()
    train_loss = 0
    metric_mae = MeanAbsoluteError().to(device)
    metric_mse = MeanSquaredError().to(device)
    metric_pcc = PearsonCorrCoef().to(device)
    metric_kcc = KendallRankCorrCoef().to(device)

    for data in train_loader:
        voxel, seq, second_struct, gt, seq_lengths = data
        # print(seq_lengths)
        out = model((voxel.to(device), seq.to(device)), seq_lengths)
        loss = criterion(out.view(-1), gt.to(device).float().view(-1))

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
            out = model((voxel.to(device), seq.to(device)), seq_lengths)
            preds.append(out)

    # calculate metrics
    preds = torch.cat(preds, dim=0)
    gt_list_valid = torch.cat(gt_list_valid).float()

    mae = metric_mae(preds.view(-1), gt_list_valid.view(-1)).item()
    mse = metric_mse(preds.view(-1), gt_list_valid.view(-1)).item()
    pcc = metric_pcc(preds.view(-1), gt_list_valid.view(-1)).item()
    kcc = metric_kcc(preds.view(-1), gt_list_valid.view(-1)).item()
    return train_loss, mae, mse, pcc, kcc
