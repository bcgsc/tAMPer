import torch
import gc
import torch_geometric.data
from loguru import logger
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.nn import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    Sigmoid)
import torch.nn.functional as F
import numpy as np
from dataset import ToxicityData
from utils import cal_metrics, check_loss, monitor_metric
import wandb
from itertools import chain


def train(model: torch.nn.Module,
          max_num_epochs: int,
          check_val_every: int,
          use_wandb: bool,
          training_loader: DataLoader,
          val_loader: DataLoader,
          embeddings: dict,
          loss_fun: dict,
          beta: float,
          warmup: int,
          optimizer: torch.optim,
          accum_iter: int,
          device: torch.device,
          monitor: str,
          checkpoint_address: str,
          patience: int,
          scheduler: torch.optim.lr_scheduler):

    log_dict = {"train_metrics": [], "val_metrics": []}

    for epoch in range(max_num_epochs):

        logger.info(f"Epoch {epoch} / {max_num_epochs}:")

        if epoch == warmup and warmup > 0:
            for param in chain(model.GRU.parameters(),
                               model.GNN.parameters(),
                               model.LayerNorm['seq'].parameters(),
                               model.LayerNorm['graph'].parameters()):
                param.requires_grad = True

        model.train()

        for iteration, graphs in enumerate(training_loader):

            sequences = [embeddings['train'][id] for id in graphs.id]

            sequences = torch.cat(sequences).to(device)
            graphs = graphs.to(device)

            preds = model(sequences, graphs)

            tx_loss = loss_fun['tx'](preds['tx'].flatten(), graphs.y)
            ss_loss = loss_fun['ss'](preds['ss'], graphs.ss.flatten())

            loss = (1 - beta) * tx_loss + beta * ss_loss

            loss = loss / accum_iter
            loss.backward()

            if ((iteration + 1) % accum_iter == 0) or (iteration + 1 == len(training_loader)):
                optimizer.step()
                optimizer.zero_grad()

        if epoch % check_val_every == 0:

            logger.info("Training metrics:")
            tr_pred, tr_metrics = check_loss(model=model,
                                             loader=training_loader,
                                             beta=beta,
                                             embeddings=embeddings['train'],
                                             device=device,
                                             loss_func=loss_fun,
                                             threshold=0.5)

            logger.info("Validation metrics:")
            val_pred, val_metrics = check_loss(model=model,
                                               loader=val_loader,
                                               beta=beta,
                                               embeddings=embeddings['val'],
                                               device=device,
                                               loss_func=loss_fun,
                                               threshold=0.5)

            tr_metrics = cal_metrics(tr_pred, tr_metrics)
            val_metrics = cal_metrics(val_pred, val_metrics)

            log_dict["train_metrics"].append(tr_metrics)
            log_dict["val_metrics"].append(val_metrics)

            if use_wandb:
                wandb.log({
                    f'train_tx_loss': log_dict["train_metrics"][-1]['tx_loss'],
                    f'train_ss_loss': log_dict["train_metrics"][-1]['ss_loss'],
                    # f'train_F1': log_dict["train_metrics"][-1]['f1'],
                    # f'train_acc': log_dict["train_metrics"][-1]['accuracy'],
                    f'val_tx_loss': log_dict["val_metrics"][-1]['tx_loss'],
                    f'val_ss_loss': log_dict["train_metrics"][-1]['ss_loss'],
                    # f'val_F1': log_dict["val_metrics"][-1]['f1'],
                    # f'val_acc': log_dict["val_metrics"][-1]['accuracy'],
                }, step=epoch)

            status = monitor_metric(log_dict=log_dict,
                                    cur_model=model,
                                    chkpnt_addr=checkpoint_address,
                                    patience=patience,
                                    metric=monitor)
            if status:
                logger.info(f"Early stopping - validation loss has"
                            f" not improved in >= {patience} times")
                return log_dict

        scheduler.step()

    return log_dict


def setup_train(config: dict,
                device: torch.device,
                tAMPer: torch.nn.Module,
                GNN: torch.nn.Module,
                tr_seqs: list,
                val_seqs: list,
                tr_pdbs: str,
                val_pdbs: str,
                pre_chkpnt: str,
                checkpoint_dir: str,
                log_file: str,
                tr_embed: str,
                val_embed: str):

    logger.info('Training set:')

    tr_data = ToxicityData(seqs_file=tr_seqs,
                           pdbs_path=tr_pdbs,
                           device=device,
                           embeddings_dir=tr_embed,
                           max_d=config['max_distance'])

    logger.info('Validation set:')

    val_data = ToxicityData(seqs_file=val_seqs,
                            pdbs_path=val_pdbs,
                            device=device,
                            embeddings_dir=val_embed,
                            max_d=config['max_distance'])

    if config['wandb']:
        wandb.init(
            project=f"{config['wdb_name']}",
            name=f"embed_{config['embedding_method']}"
                 f"_beta_{config['beta']}",
            config={'architecture': 'gru+gnn+att'})

    train_dl = DataLoader(
        dataset=tr_data,
        batch_size=config["batch_size"],
        num_workers=8,
        shuffle=True)

    val_dl = DataLoader(
        dataset=val_data,
        batch_size=config["batch_size"],
        num_workers=8,
        shuffle=True)

    loss_function = {'tx': BCEWithLogitsLoss(),
                     'ss': CrossEntropyLoss()}

    tAMPer.to(device=device)

    if pre_chkpnt != 'None':
        logger.info(f"Loading the pre-trained model from {pre_chkpnt}/chkpnt.pt")
        GNN.load_state_dict(torch.load(f'{pre_chkpnt}/chkpnt.pt')['model'])
        logger.info(f"Transferring weights to tAMPer")
        tAMPer.GNN.load_state_dict(GNN.layers.state_dict())

    del GNN
    torch.cuda.empty_cache()
    gc.collect()

    if config['modality'] == 'all' and config['warmup'] > 0:
        for param in chain(tAMPer.GRU.parameters(),
                           tAMPer.GNN.parameters(),
                           tAMPer.LayerNorm.parameters()):
            param.requires_grad = False

    optimizer = Adam(tAMPer.parameters(),
                     lr=config["optimizer_lr"],
                     weight_decay=config["optimizer_weight_decay"])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                gamma=0.5,
                                                step_size=config["step"])
    logger.info("Training:")

    log = train(
        model=tAMPer,
        accum_iter=config["accum_iter"],
        max_num_epochs=config["max_epochs"],
        check_val_every=config["val_every"],
        training_loader=train_dl,
        val_loader=val_dl,
        embeddings={'train': tr_data.seq_embeddings,
                    'val': val_data.seq_embeddings},
        loss_fun=loss_function,
        beta=config['beta'],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_wandb=config['wandb'],
        patience=config["patience"],
        warmup=config['warmup'],
        monitor=config['monitor'],
        checkpoint_address=checkpoint_dir)

    np.save(log_file, log)

    return log
