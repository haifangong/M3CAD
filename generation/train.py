import time

import torch
import torch.nn.functional as F

from loss import compute_ce_loss, compute_mmd, kl_gaussianprior


def train(args, epoch, model, train_loader, device, optimizer):
    """
    Train the generation model (VAE or WAE) for one epoch.

    Args:
        args: Command line arguments containing model configuration.
        epoch (int): Current epoch number.
        model: The PyTorch model to train (SEQVAE, MMVAE, etc.).
        train_loader: DataLoader for training data.
        device: Torch device (CPU/GPU).
        optimizer: Optimizer instance.

    Returns:
        tuple: (Total Loss, (Voxel Loss, Seq Loss, Gen Loss, Cls Loss))
    """
    model.train()
    loss_vox = 0.0
    loss_seq = 0.0
    loss_gen = 0.0
    loss_cls = 0.0
    yt = []
    start_time = time.time()

    for data in train_loader:
        voxel, seq, gt = data
        voxel = voxel.to(device)
        seq = seq.to(device)
        gt = gt.to(device)
        yt.append(gt)

        # Forward pass based on model type (VAE vs WAE) and architecture (Sequence vs Multimodal)
        if args.gen_model == 'vae':
            if args.model == 'seq':
                seq_rec, classify_result, mean, var, cond_f = model((voxel, seq), gt)
                vox_rec = None  # Voxel reconstruction not used in sequence-only model
            else:
                (vox_rec, seq_rec), classify_result, mean, var, cond_f = model((voxel, seq), gt)

        elif args.gen_model == 'wae':
            if args.model == 'seq':
                seq_rec, classify_result, z = model((voxel, seq), gt)
                vox_rec = None  # Voxel reconstruction not used in sequence-only model
            else:
                (vox_rec, seq_rec), classify_result, z = model((voxel, seq), gt)
        else:
            raise ValueError(f"Unsupported gen_model '{args.gen_model}'")

        # --- Loss Calculation ---

        # 1. Voxel Reconstruction Loss (Smooth L1)
        if vox_rec is not None:
            loss_rec_vox = F.smooth_l1_loss(vox_rec, voxel, beta=0.01, reduction='mean')
        else:
            loss_rec_vox = torch.tensor(0)

        # 2. Sequence Reconstruction Loss (Cross Entropy)
        loss_rec_seq = compute_ce_loss(seq_rec, seq)

        # 3. Generative Loss (KLD for VAE, MMD for WAE)
        if args.gen_model == 'vae':
            loss_norm = kl_gaussianprior(mean, var)
        elif args.gen_model == 'wae':
            z_fake = torch.randn_like(z).to(device)
            mmd_loss = compute_mmd(z, z_fake)
            loss_norm = mmd_loss
        else:
            loss_norm = torch.tensor(0.0, device=device)

        # Combine losses
        if loss_rec_vox.item() != 0:
            loss = loss_rec_vox + loss_rec_seq + loss_norm
        else:
            loss = loss_rec_seq + loss_norm

        # 4. Classification Loss (Optional Supervised Loss)
        if args.sup:
            loss_cls_value = F.mse_loss(classify_result, gt)
            loss += loss_cls_value
        else:
            loss_cls_value = 0
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
        optimizer.step()

        # Record losses for logging
        loss_vox += loss_rec_vox.item()
        loss_seq += loss_rec_seq.item()
        loss_gen += loss_norm.item()
        loss_cls += loss_cls_value.item()

    # Calculate average losses for the epoch
    loss_vox /= len(train_loader)
    loss_seq /= len(train_loader)
    loss_gen /= len(train_loader)
    loss_cls /= len(train_loader)

    epoch_time = time.time() - start_time
    loss_sum = loss_vox + loss_seq + loss_gen
    print(f'Epoch: {epoch} Time: {epoch_time:.1f}s Loss: {loss_sum:.1f}; Voxel {loss_vox:.1f}; Seq {loss_seq:.1f}; Gen {loss_gen:.1f}; Cls {loss_cls:.1f}')

    return loss_sum, (loss_vox, loss_seq, loss_gen, loss_cls)
