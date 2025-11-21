from typing import Tuple, List, Dict
from numpy.typing import ArrayLike

import numpy as np
import torch

from torch import nn
from torch.amp.grad_scaler import GradScaler

from .regularizers import (
    get_attention_activations,
    regularize_attention_weights,
    _graph_trend_filtering,
)

from ..layers.task_utils import (
    convert_functional_map_to_pointwise_map,
    geodesic_label_errors,
)

try:
    import wandb
except ImportError:
    print("`wandb` was not found. `wandb` functionalities will be disabled.")


def train_epoch(
    data_loader, model, loss_compute, optimizer, scheduler, epoch, device, training_kw
):
    """
    Trains model for one epoch and logs training stats.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        A `pytorch` dataloader.
    model : torch.nn.Module,
        A pytorch model.
    loss_compute : torch.nn.loss
        A torch loss calculation instance.
    optimizer : torch.optim
        A `pytorch` optimizer.
    scheduler : torch.optim.lr_scheduler
        A `pytorch` learning rate scheduler.
    epoch : int
        Current training epoch.
    device : torch.device
        Device to use, i.e. cuda or cpu.
    training_kw : Dict
        Model training keyword arguments.

    Returns
    -------
    None
        Performs model training
    """
    # switch into training mode and initialize statistics
    model.train()

    # if training_kw["use_reg"] is True and training_kw["reg_type"] == "activations":
    #     outputs = get_attention_activations(model)

    epochs = training_kw["epochs"]
    task = training_kw["task"]

    train_metrics = {
        "corrects": 0,
        "acc": 0.0,
        "loss": 0.0,
    }

    scaler = GradScaler()

    total = 0

    all_preds, all_trues = [], []

    for idx, (tensors, label, weight) in enumerate(data_loader):

        # zero gradients
        optimizer.zero_grad()

        with torch.autocast(device.type, dtype=torch.bfloat16):
            # mount data to device
            tensors = [tensor.to(device.type) for tensor in tensors]

            label = label.to(device.type)
            weight = weight.to(device.type)

            # make predictions, compute loss, compute gradients, update weights
            logits = model(*tensors, mask=None)
            loss = loss_compute(logits, label.squeeze(1))

        # if training_kw["use_reg"] is True:
        #
        #     _lambda = training_kw["_lambda"]
        #
        #     if training_kw["reg_type"] == "activations":
        #
        #         # Regularize with trend filtering
        #         d_0, d_1 = tensors[3], tensors[4]
        #
        #         for space, activations in outputs.items():
        #
        #             if space == "v_linears":
        #                 loss += _graph_trend_filtering(activations, d_0, _lambda)
        #
        #             if space == "e_linears":
        #                 loss += _graph_trend_filtering(activations, d_1, _lambda)
        #
        #     if training_kw["reg_type"] == "weights":
        #
        #         Regularize with lasso
        #         loss += regularize_attention_weights(model, _lambda=l1_lambda)

        # calculate gradient. Use scaler if device == "cuda"
        if device.type == "cpu":
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

            scaler.step(optimizer)
            scaler.update()

        # log metrics
        if task == "classification":
            preds, trues = log_multiclass__classification(logits, label)

        elif task == "segmentation":
            preds, trues = log_multiclass__segmentation(logits, label)

        total += (trues != -1).sum()

        train_metrics = log_update_metrics(
            train_metrics, preds, trues, loss, idx, total
        )

        all_trues.append(trues)
        all_preds.append(preds)

        print(
            "[TRAIN] [EPOCH:{}/{} ] [IDX: {}/{}] Loss: {:.4f} Accuracy: {:.4f}\t".format(
                epoch, epochs, idx + 1, "UNK", loss.item(), train_metrics["acc"]
            )
        )

    train_metrics["loss"] = train_metrics["loss"] / idx

    try:
        _log = {"train_" + k: v for k, v in train_metrics.items()}

        # if task == "segmentation":
        #     _log.update(
        #         wandb_confusion_matrix_segmentation(all_trues, all_preds, "train_")
        #     )

        wandb.log(_log, step=epoch)

    except Exception:
        pass

    scheduler.step()

    return train_metrics


def validate_epoch(data_loader, model, loss_compute, epoch, device, training_kw):
    """
    Validates the model after each epoch and logs the metrics.

    Parameters
    ----------
    training_kw : Dict
        Model training keyword arguments.
    data_loader : torch.utils.data.DataLoader
        A `pytorch` dataloader.
    model : torch.nn.Models
        A `pytorch` model.
    epoch : int
        Current epoch.
    device : torch.device
        Device to validate on i.e. 'cpu' or 'cuda'.

    Returns
    -------
    None
        Performs model validation. Logs validation metrics to `wandb` logger.
    """
    # switch into eval mode
    model.eval()

    val_metrics = {
        "corrects": 0,
        "acc": 0.0,
        "loss": 0.0,
    }

    epochs = training_kw["epochs"]
    task = training_kw["task"]
    total = 0

    all_trues, all_preds = [], []

    for idx, (tensors, label, weight) in enumerate(data_loader):

        # zero gradients
        with torch.no_grad():

            with torch.autocast(device.type, dtype=torch.bfloat16):
                # mount data to device
                tensors = [tensor.to(device.type) for tensor in tensors]

                label = label.to(device.type)
                weight = weight.to(device.type)

                # make predictions, compute loss, compute gradients, update weights
                logits = model(*tensors, mask=None)
                loss = loss_compute(logits, label.squeeze(1))

            # log metrics
            if task == "classification":
                preds, trues = log_multiclass__classification(logits, label)

            elif task == "segmentation":
                preds, trues = log_multiclass__segmentation(logits, label)

            total += (trues != -1).sum()

            val_metrics = log_update_metrics(
                val_metrics, preds, trues, loss, idx, total
            )

            print(
                "[VALID] [EPOCH:{}/{} ] [IDX: {}/{}] Loss: {:.4f} Accuracy: {:.4f}\t".format(
                    epoch, epochs, idx + 1, "UNK", loss.item(), val_metrics["acc"]
                )
            )

        all_trues.append(trues)
        all_preds.append(preds)

    val_metrics["loss"] = val_metrics["loss"] / idx

    try:
        _log = {"val_" + k: v for k, v in val_metrics.items()}

        if task == "segmentation":
            _log.update(
                wandb_confusion_matrix_segmentation(all_trues, all_preds, "val_")
            )

        wandb.log(_log, step=epoch)

    except Exception:
        pass

    return val_metrics


def train_epoch_fc(
    data_loader, model, loss_compute, optimizer, scheduler, epoch, device, training_kw
):
    """
    Trains model for one epoch and logs training stats.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        A `pytorch` dataloader.
    model : torch.nn.Module,
        A pytorch model.
    loss_compute : torch.nn.loss
        A torch loss calculation instance.
    optimizer : torch.optim
        A `pytorch` optimizer.
    scheduler : torch.optim.lr_scheduler
        A `pytorch` learning rate scheduler.
    epoch : int
        Current training epoch.
    device : torch.device
        Device to use, i.e. cuda or cpu.
    training_kw : Dict
        Model training keyword arguments.

    Returns
    -------
    None
        Performs model training
    """
    # switch into training mode and initialize statistics
    model.train()

    epochs = training_kw["epochs"]

    train_metrics = {
        "loss": 0.0,
    }

    scaler = GradScaler()

    for idx, (tensors_s, tensors_t, C_gt, evecvals_s, evecvals_t, i_idx) in enumerate(
        data_loader
    ):

        # zero gradients
        optimizer.zero_grad()

        with torch.autocast(device.type, dtype=torch.bfloat16):
            # mount data to device
            tensors_s = [tensor.to(device.type) for tensor in tensors_s]
            tensors_t = [tensor.to(device.type) for tensor in tensors_t]

            C_gt = C_gt.to(device.type)

            evecs_s, evals_s = [tensor.to(device.type) for tensor in evecvals_s[:2]]
            evecs_t, evals_t = [tensor.to(device.type) for tensor in evecvals_t[:2]]

            # make predictions, compute loss, compute gradients, update weights
            C_pred, _, _ = model(
                tensors_s, tensors_t, evecs_s, evecs_t, evals_s, evals_t, mask=None
            )

            loss = loss_compute(C_pred, C_gt)

        # calculate gradient. Use scaler if device == "cuda"
        if device.type == "cpu":
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

            scaler.step(optimizer)
            scaler.update()

        print(
            "[TRAIN] [EPOCH:{}/{} ] [IDX: {}/{}] Loss: {:.4f}\t".format(
                epoch, epochs, idx + 1, "UNK", loss.item()
            )
        )

        train_metrics["loss"] += loss.item()

    train_metrics["loss"] = train_metrics["loss"] / idx

    try:
        _log = {"train_" + k: v for k, v in train_metrics.items()}

        wandb.log(_log, step=epoch)

    except Exception:
        pass

    scheduler.step()

    return train_metrics


def validate_epoch_fc(data_loader, model, loss_compute, epoch, device, training_kw):
    """
    Validates the model after each epoch and logs the metrics.

    Parameters
    ----------
    training_kw : Dict
        Model training keyword arguments.
    data_loader : torch.utils.data.DataLoader
        A `pytorch` dataloader.
    model : torch.nn.Models
        A `pytorch` model.
    epoch : int
        Current epoch.
    device : torch.device
        Device to validate on i.e. 'cpu' or 'cuda'.

    Returns
    -------
    None
        Performs model validation. Logs validation metrics to `wandb` logger.
    """
    # switch into eval mode
    model.eval()

    val_metrics = {
        "loss": 0.0,
        "geodesic_error": 0.0,
    }

    epochs = training_kw["epochs"]

    estimate_geodesic_error = True

    for idx, (tensors_s, tensors_t, C_gt, evecvals_s, evecvals_t, i_idx) in enumerate(
        data_loader
    ):

        # zero gradients
        with torch.no_grad():

            with torch.autocast(device.type, dtype=torch.bfloat16):
                # mount data to device
                tensors_s = [tensor.to(device.type) for tensor in tensors_s]
                tensors_t = [tensor.to(device.type) for tensor in tensors_t]

                C_gt = C_gt.to(device.type)

                evecs_s, evals_s = [tensor.to(device.type) for tensor in evecvals_s[:2]]
                evecs_t, evals_t = [tensor.to(device.type) for tensor in evecvals_t[:2]]

                # make predictions, compute loss, compute gradients, update weights
                C_pred, _, _ = model(
                    tensors_s, tensors_t, evecs_s, evecs_t, evals_s, evals_t, mask=None
                )
                loss = loss_compute(C_pred, C_gt)

            print(
                "[VALID] [EPOCH:{}/{} ] [IDX: {}/{}] Loss: {:.4f}\t".format(
                    epoch, epochs, idx + 1, "UNK", loss.item()
                )
            )

        val_metrics["loss"] += loss.item()

        b = i_idx.size(0)

        if estimate_geodesic_error is False:
            continue

        for b_idx in range(b):

            _i = i_idx[b_idx].item()

            geo_dist_path_s, _ = data_loader.dataset.paths_geodists[_i]

            npzfile = np.load(geo_dist_path_s, allow_pickle=True)

            G_s = npzfile["dist"]

            n_s = tensors_s[0][b_idx].size(0)

            f_area_s = tensors_s[2][b_idx][:, -1].numpy(force=True)

            vts_s = evecvals_s[2][b_idx].numpy(force=True)
            vts_t = evecvals_t[2][b_idx].numpy(force=True)

            _C_pred = C_pred[b_idx].float().numpy(force=True)

            _evecs_s = evecs_s[b_idx].numpy(force=True)
            _evecs_t = evecs_t[b_idx].numpy(force=True)

            T_t_to_s = convert_functional_map_to_pointwise_map(
                _C_pred, _evecs_s, _evecs_t
            )

            T_t_to_s = T_t_to_s[vts_t]

            errors = geodesic_label_errors(
                G_s,
                vts_pred=T_t_to_s,
                vts_true=vts_s,
                area=f_area_s.sum(),
                normalization="area",
            )

            # pmap = T_t_to_s

            # ind21 = np.stack([vts_s, pmap[vts_t]], axis=-1)
            # ind21 = np.ravel_multi_index(ind21.T, dims=[n_s, n_s])

            # Zoom Out
            # T21_ref, C_ref = refine_pMap_zo(T21, ev_t, ev_s, n_eig)
            #
            # pmap = T21_ref
            # ind21_ref = np.stack([phi_s, pmap[phi_t]], axis=-hp1)
            # ind21_ref = np.ravel_multi_index(ind21_ref.T, dims = [n_s, n_s])

            # errs = np.take(G_s, ind21) / SQ_s
            # errs_ref = np.take(G_s, ind21_ref) / SQ_s

        val_metrics["geodesic_error"] += np.mean(errors)

    val_metrics["loss"] = val_metrics["loss"] / idx
    val_metrics["geodesic_error"] = val_metrics["geodesic_error"] / idx

    try:
        _log = {"val_" + k: v for k, v in val_metrics.items()}

        wandb.log(_log, step=epoch)

    except Exception:
        pass

    return val_metrics


def infer_epoch(data_loader, model, device, kw):
    """
    Validates the model after each epoch and logs the metrics.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        A `pytorch` dataloader.
    model : torch.nn.Models
        A `pytorch` model.
    device : torch.device
        Device to validate on i.e. 'cpu' or 'cuda'.
    kw : Dict
        Model training keyword arguments.

    Returns
    -------
    all_preds
        Model predictions for input samples.
    """
    # switch into eval mode
    model.eval()

    task = kw["task"]

    all_preds = []

    for idx, tensors in enumerate(data_loader):

        # zero gradients
        with torch.no_grad():

            with torch.autocast(device.type, dtype=torch.bfloat16):
                # mount data to device
                tensors = [tensor.to(device.type) for tensor in tensors]

                logits = model(*tensors, mask=None)

            # log metrics
            if task == "classification":
                preds, _ = log_multiclass__classification(logits, label=None)

            elif task == "segmentation":
                preds, _ = log_multiclass__segmentation(logits, label=None)

        all_preds.append(preds.tolist())

    return [pred for preds in all_preds for pred in preds]


def log_multiclass__classification(
    logits: torch.Tensor, label: torch.Tensor | None
) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Prepare predictions for metric calculation.
    """
    # extract probabilities and measure metrics
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)

    if label is None:
        return preds.numpy(force=True), None

    trues = label.squeeze(1)

    return preds.numpy(force=True), trues.numpy(force=True)


def log_multiclass__segmentation(
    logits: torch.Tensor, label: torch.Tensor | None
) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Prepare predictions for metric calculation.
    """
    # extract probabilities and measure metrics
    probs = torch.softmax(logits, dim=-2)
    preds = torch.argmax(probs, dim=-2)

    if label is None:
        return preds.numpy(force=True), None

    trues = label.squeeze(1)

    return preds.numpy(force=True), trues.numpy(force=True)


def log_update_metrics(
    metrics: Dict,
    preds: ArrayLike,
    trues: ArrayLike,
    loss: torch.Tensor,
    idx: int,
    total: int,
) -> Dict:
    """
    Log classification metrics.
    """
    metrics["corrects"] += (preds == trues).sum()
    metrics["acc"] = metrics["corrects"] / total
    metrics["loss"] += loss.item()

    return metrics


def wandb_confusion_matrix_segmentation(all_trues, all_preds, prefix=""):
    """
    Create a wandb confusion matrix
    """
    KEY = prefix + "conf_mat"

    return {
        KEY: wandb.plot.confusion_matrix(
            probs=None,
            y_true=np.concatenate([record for batch in all_trues for record in batch]),
            preds=np.concatenate([record for batch in all_preds for record in batch]),
        )
    }


def model_save(model, optimizer, scheduler, epoch, save_path):
    """
    Saves model, optimizer and scheduler history at some specific epoch.

    Parameters
    ----------
    model : torch.nn.Module.
        An instantiation of a pytorch model.
    optimizer : torch.optim
        A `pytorch` optimizer.
    scheduler : torch.optim.lr_scheduler
        A `pytorch` learning rate scheduler.
    epoch : int
        Current epoch in training loop.
    save_path : str
        Save path to store model.

    Returns
    -------
    None
        Saves model state along with optimizer, scheduler history etc.
    """
    save_dict = {
        "state": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
    }

    torch.save(save_dict, save_path)


def validate_device(device):
    """
    Validate device argument.
    """
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    elif device == "xpu" and hasattr(torch, "xpu"):
        return torch.device("xpu" if torch.xpu.is_available() else "cpu")

    else:
        return torch.device("cpu")
