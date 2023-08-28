import torch
import numpy as np
import pandas as pd
import os
from argparse import ArgumentParser
import torch_geometric.data
from loguru import logger
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.nn import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    Sigmoid)

from dataset import ToxicityData
from utils import cal_metrics, check_loss, monitor_metric, set_seed, retrieve_best_model
from tAMPer import tAMPer


def train(model: torch.nn.Module,
          max_num_epochs: int,
          check_val_every: int,
          training_loader: DataLoader,
          val_loader: DataLoader,
          embeddings: dict,
          loss_fun: dict,
          beta: float,
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


def setup_train():

    parser = ArgumentParser()
    # data
    parser.add_argument('-tr_pos', '--tr_pos', default=f'{os.getcwd()}/data/sequences/tr_pos.faa', type=str,
                        required=True)
    parser.add_argument('-tr_neg', '--tr_neg', default=f'{os.getcwd()}/data/sequences/tr_neg.faa', type=str,
                        required=True)
    parser.add_argument('-pdb_dir', '--pdb_dir', default=f'{os.getcwd()}/data/structures/', type=str,
                        required=True)
    parser.add_argument('-val_pos', '--val_pos', default=f'{os.getcwd()}/data/sequences/val_pos.faa',
                        type=str, required=True)
    parser.add_argument('-val_neg', '--val_neg', default=f'{os.getcwd()}/data/sequences/val_neg.faa',
                        type=str, required=True)
    # model configs
    parser.add_argument('-lr', '--learning_rate', default=0.0004, type=float, required=False)
    parser.add_argument('-hdim', '--hdim', default=32, type=int, required=False)
    parser.add_argument('-sn', '--sequence_num_layers', default=1, type=int, required=False)
    parser.add_argument('-emd', '--embedding', default="t6", type=str, required=False)
    parser.add_argument('-gl', '--gnn_layers', default=3, type=int, required=False)
    parser.add_argument('-bz', '--batch_size', default=32, type=int, required=False)
    parser.add_argument('-eph', '--num_epoch', default=50, type=int, required=False)
    parser.add_argument('-acg', '--accum_iter', default=1, type=int, required=False)
    parser.add_argument('-wd', '--weight_decay', default=1e-7, type=float, required=False)
    parser.add_argument('-sqd', '--seq_dropout', default=0.0, type=float, required=False)
    parser.add_argument('-std', '--strct_dropout', default=0.0, type=float, required=False)
    parser.add_argument('-mdl', '--modality', default="all", type=str, required=False)
    parser.add_argument('-dm', '--d_max', default=10, type=int, required=False)
    parser.add_argument('-beta', '--beta', default=0.0, type=float, required=False)
    parser.add_argument('-monitor', '--monitor_meter', default='loss', type=str, required=False)
    parser.add_argument('-step', '--step', default=50, type=int, required=False)
    parser.add_argument('-seed', '--seed', default=1, type=int, required=False)
    # addresses
    parser.add_argument('-pck', '--pre_chkpnt', default='', type=str, required=False)
    parser.add_argument('-ck', '--checkpoint_dir', default=f'{os.getcwd()}/checkpoints/chkpnt.pt',
                        type=str, required=False)
    parser.add_argument('-log', '--log_file', default=f'{os.getcwd()}/logs/log.npy', type=str, required=False)
    parser.add_argument('-res', '--res', default=f'{os.getcwd()}/results/val.csv', type=str, required=False)

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.embedding == 't6':
        seq_inp_dim = 320
    elif args.embedding == 't12':
        seq_inp_dim = 480
    elif args.embedding == 't30':
        seq_inp_dim = 640
    else:
        seq_inp_dim = 1280

    model = tAMPer(
        input_modality=args.modality,
        seq_input_dim=seq_inp_dim,
        douts={'seq': args.seq_dropout, 'strct': args.strct_dropout},
        node_dims=(6, 3),
        edge_dims=(32, 1),
        node_h_dim=(args.hdim, 16),
        edge_h_dim=(32, 1),
        gru_hidden_dim=int(args.hdim / 2),  # bi-directional
        gru_layers=args.sequence_num_layers,
        num_gnn_layers=args.gnn_layers)

    logger.info('Training set:')

    tr_data = ToxicityData(seqs_file=[args.tr_neg, args.tr_pos],
                           pdbs_path=args.pdb_dir,
                           device=device,
                           embeddings_dir=tr_embed,
                           max_d=args.d_max)

    logger.info('Validation set:')

    val_data = ToxicityData(seqs_file=[args.val_neg, args.val_pos],
                            pdbs_path=args.pdb_dir,
                            device=device,
                            embeddings_dir=val_embed,
                            max_d=args.d_max)

    train_dl = DataLoader(
        dataset=tr_data,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True)

    val_dl = DataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True)

    loss_function = {'tx': BCEWithLogitsLoss(),
                     'ss': CrossEntropyLoss()}

    optimizer = Adam(model.parameters(),
                     lr=args.learning_rate,
                     weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                gamma=0.5,
                                                step_size=args.step)

    if pre_chkpnt:
        logger.info(f"Loading the pre-trained model from {args.pre_chkpnt}")
        model.GNN.load_state_dict(torch.load(args.pre_chkpnt)['model'].layers, strict=True)

    model.to(device=device)

    logger.info("Training:")

    log = train(
        model=model,
        accum_iter=args.accum_iter,
        max_num_epochs=args.num_epoch,
        check_val_every=1,
        training_loader=train_dl,
        val_loader=val_dl,
        embeddings={'train': tr_data.seq_embeddings,
                    'val': val_data.seq_embeddings},
        loss_fun=loss_function,
        beta=args.beta,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        patience=args.patience,
        monitor=args.monitor_meter,
        checkpoint_address=checkpoint_dir)

    np.save(log_file, log)
