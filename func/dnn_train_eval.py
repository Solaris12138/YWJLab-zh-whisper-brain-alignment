import os
import copy
import torch
import numpy as np

from tqdm import tqdm
from logzero import logger


def reconstruct_loss(truth_signal, recon_signal, reg_lambda = 0.5):
    loss_fn_1 = torch.nn.MSELoss()
    loss_fn_2 = torch.nn.SmoothL1Loss()
    return reg_lambda * loss_fn_1(recon_signal, truth_signal) + (1 - reg_lambda) * loss_fn_2(recon_signal, truth_signal)


def train_model(model, train_loader, val_loader, save_dir, fname,
                num_epochs=100, learning_rate=0.001,
                weight_decay=1e-4, patience=10, min_delta=1e-4, device="cpu",):
    """
    Train a DNN-based encoding model.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    train_loader : torch.utils.data.DataLoader
        Dataloader for training set.
    val_loader : torch.utils.data.DataLoader
        Dataloader for validation set.
    save_dir : str | Path
        Directory to store the model parameters.
    fname : str
        File name of saved model parameters.
    num_epochs : int
        The maximum training epochs.
    learning_rate : float
        The learning rate.
    weight_decay : float
        L2 regularization coefficient.
    patience : int
        The number of epochs after which the training procedures would be stopped if the model performance
        did not improve on the validation set.
    min_delta : float
        Threshold for model performance improvement.
    
    Returns
    -------
    NoneType

    """

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience//2, factor=0.5, verbose=True)

    best_loss = np.inf
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch_idx in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for _, batch_data in tqdm(enumerate(train_loader), total=len(train_loader), 
                                          desc=f"Epoch {epoch_idx}", ncols=80, leave=False, dynamic_ncols=False):
            batch_embedding, batch_meg = batch_data[0], batch_data[1]

            batch_embedding = batch_embedding.to(device, non_blocking=True)
            batch_meg = batch_meg.to(device, non_blocking=True)

            optimizer.zero_grad()
            meg_reconstructed = model(batch_embedding)
            loss = reconstruct_loss(batch_meg, meg_reconstructed, reg_lambda=1)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        if_valid = (epoch_idx >= 0)
        if if_valid:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _, batch_data in tqdm(enumerate(val_loader), total=len(val_loader), 
                                                  ncols=80, leave=False, dynamic_ncols=False):
                    val_embedding, val_meg = batch_data[0], batch_data[1]
                    val_embedding = val_embedding.to(device, non_blocking=True)
                    val_meg = val_meg.to(device, non_blocking=True)
                    val_recon = model(val_embedding)
                    loss = reconstruct_loss(val_meg, val_recon, reg_lambda=1)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            scheduler.step(avg_val_loss)

            if avg_val_loss + min_delta < best_loss:
                best_loss = avg_val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                torch.save(best_model_wts, os.path.join(save_dir, fname))
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break
        else:
            scheduler.step(avg_train_loss)
        
        if if_valid:
            logger.info(f"Epoch: {epoch_idx} "
                        f"train loss: {avg_train_loss:.4f} "
                        f"valid loss: {avg_val_loss:.4f}")
        else:
            logger.info(f"Epoch: {epoch_idx} "
                        f"train loss: {avg_train_loss:.4f} "
                        "valid loss: N/A")

    model.load_state_dict(best_model_wts)


def evaluate_model(model, test_loader, device="cpu"):
    """
    Test a DNN-based encoding model.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    test_loader : torch.utils.data.DataLoader
        Dataloader for test set.
    
    Returns
    -------
    scores : numpy.ndarray. Shape: (n_vertices, )
        Pearson's r-value between predicted signals and ground truth for all vertices.

    """
    model.eval()
    
    total_loss = 0.0
    all_recon = []
    all_truth = []
    
    with torch.no_grad():
        for batch_idx, batch_data in tqdm(enumerate(test_loader), total=len(test_loader), 
                                          desc=f"Epoch Test", ncols=80, leave=False, dynamic_ncols=False):
            batch_embedding, batch_meg = batch_data[0], batch_data[1]

            batch_embedding = batch_embedding.to(device)
            batch_meg = batch_meg.to(device)
            
            meg_recon = model(batch_embedding)
            
            loss = reconstruct_loss(batch_meg, meg_recon, reg_lambda=1)
            total_loss += loss.item()

            all_recon.append(meg_recon.cpu().numpy())
            all_truth.append(batch_meg.cpu().numpy())
            
    avg_test_loss = total_loss / len(test_loader)
    logger.info(f"Average Test Loss: {avg_test_loss:.4f}")
    
    all_recon = np.concatenate(all_recon, axis=0)
    all_truth = np.concatenate(all_truth, axis=0)

    n_sginals = all_recon.shape[-1]
    scores = np.zeros(n_sginals)
    for idx in range(n_sginals):
        recon_concatenated = all_recon[:, :, idx].reshape(-1)
        truth_concatenated = all_truth[:, :, idx].reshape(-1)
        scores[idx] = np.corrcoef(recon_concatenated, truth_concatenated)[0, 1]

    return scores