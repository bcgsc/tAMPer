import torch
import numpy as np
import os
from argparse import ArgumentParser
from loguru import logger
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.nn import (
    BCEWithLogitsLoss,
    CrossEntropyLoss, )

from dataset import SimpleData
from utils import cal_metrics, check_loss, monitor_metric, set_seed, calculate_pos_weight
from tAMPer import tAMPer


def train(model: tAMPer,
          max_num_epochs: int,
          check_val_every: int,
          training_loader: DataLoader,
          val_loader: DataLoader,
          loss_fun: dict,
          lammy: float,
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

            graphs = graphs.to(device)
            embeddings = graphs.embeddings

            preds = model(embeddings, graphs)

            tx_loss = loss_fun['tx'](preds['tx'].flatten(), graphs.y)
            ss_loss = loss_fun['ss'](preds['ss'], graphs.ss.flatten())

            loss = (1 - lammy) * tx_loss + lammy * ss_loss

            loss = loss / accum_iter
            loss.backward()

            if ((iteration + 1) % accum_iter == 0) or (iteration + 1 == len(training_loader)):
                optimizer.step()
                optimizer.zero_grad()

        if epoch % check_val_every == 0:

            logger.info("Training metrics:")
            tr_pred, tr_losses = check_loss(model=model,
                                            loader=training_loader,
                                            lammy=lammy,
                                            device=device,
                                            loss_func=loss_fun,
                                            threshold=0.5)

            logger.info("Validation metrics:")
            val_pred, val_losses = check_loss(model=model,
                                              loader=val_loader,
                                              lammy=lammy,
                                              device=device,
                                              loss_func=loss_fun,
                                              threshold=0.5)

            tr_metrics = cal_metrics(tr_pred, tr_losses)
            val_metrics = cal_metrics(val_pred, val_losses)

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
    parser = ArgumentParser(description='train.py script runs tAMPer for training.')
    # data
    parser.add_argument('-tr_pos', default=f'{os.getcwd()}/data/sequences/tr_pos.faa', type=str,
                        required=True, help='training toxic sequences fasta file (.fasta)')

    parser.add_argument('-tr_neg', default=f'{os.getcwd()}/data/sequences/tr_neg.faa', type=str,
                        required=True, help='training non-toxic sequences fasta file (.fasta)')

    parser.add_argument('-tr_pdb', default=f'{os.getcwd()}/data/structures/', type=str,
                        required=True, help='address directory of train structures')

    parser.add_argument('-tr_embed', default=f'{os.getcwd()}/data/embeddings/', type=str,
                        required=True, help='address directory of train embeddings')

    parser.add_argument('-val_pos', default=f'{os.getcwd()}/data/sequences/val_pos.faa',
                        type=str, required=True, help='validation toxic sequences fasta file (.fasta)')

    parser.add_argument('-val_neg', default=f'{os.getcwd()}/data/sequences/val_neg.faa',
                        type=str, required=True, help='validation non-toxic sequences fasta file (.fasta)')

    parser.add_argument('-val_pdb', default=f'{os.getcwd()}/data/structures/', type=str,
                        required=True, help='address directory of val structures')

    parser.add_argument('-val_embed', default=f'{os.getcwd()}/data/embeddings/', type=str,
                        required=True, help='address directory of val embeddings')
    # model configs
    parser.add_argument('-lr', default=0.0004, type=float, required=False, help='learning rate')

    parser.add_argument('-hdim', default=32, type=int, required=False,
                        help='hidden dimension of model for h_seq and h_strct')

    parser.add_argument('-gru_layers', default=1, type=int, required=False,
                        help='number of GRU Layers')

    parser.add_argument('-embedding_model', default="t12", type=str, required=False,
                        help='different variant of ESM2 embeddings: {t6, t12, t30, t33, t36, t48}')

    parser.add_argument('-modality', default='all', type=str, required=False, help='Used modality')

    parser.add_argument('-gnn_layers', default=1, type=int, required=False, help='number of GNNs Layers')

    parser.add_argument('-batch_size', default=32, type=int, required=False, help='batch size')

    parser.add_argument('-n_epochs', default=50, type=int, required=False, help='max number of epochs')

    parser.add_argument('-gard_acc', default=1, type=int, required=False, help='gradient accumulation steps')

    parser.add_argument('-weight_decay', default=1e-7, type=float, required=False, help='weight decay')

    parser.add_argument('-dmax', default=12, type=int, required=False,
                        help='max distance to consider two connect two residues in the graph')

    parser.add_argument('-lammy', default=0.0, type=float, required=False,
                        help='lammy in the objective function')

    parser.add_argument('-monitor', default='f1', type=str, required=False,
                        help='the metric to monitor for early stopping during training')
    # addresses
    parser.add_argument('-pre_chkpnt', default='', type=str, required=False,
                        help='address of pre-trained GNNs')

    parser.add_argument('-chkpnt', default=f'{os.getcwd()}/checkpoints/chkpnt.pt',
                        type=str, required=False, help='address to where trained model be stored')

    parser.add_argument('-log', default=f'{os.getcwd()}/logs/log.npy', type=str, required=False,
                        help='address to where log file be stored')

    args = parser.parse_args()
    set_seed(1)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.embedding_model == 't6':
        seq_inp_dim = 320
    elif args.embedding_model == 't12':
        seq_inp_dim = 480
    elif args.embedding_model == 't30':
        seq_inp_dim = 640
    else:
        seq_inp_dim = 1280

    model = tAMPer(
        input_modality=args.modality,
        seq_input_dim=seq_inp_dim,
        node_dims=(6, 3),
        edge_dims=(32, 1),
        node_hdim=(args.hdim, 16),
        edge_hdim=(32, 1),
        n_heads=args.n_heads,
        gru_hdim=args.hdim,
        n_grus=args.gru_layers,
        n_gnns=args.gnn_layers)

    logger.info('Training set:')

    tr_data = ToxicityData(pos_seqs=args.tr_pos,
                           neg_seqs=args.tr_neg,
                           pdbs_path=args.pdb_dir,
                           embedding_model=args.embedding_model,
                           max_d=args.d_max)

    logger.info('Validation set:')

    val_data = ToxicityData(pos_seqs=args.val_pos,
                            neg_seqs=args.val_neg,
                            pdbs_path=args.val_pdb,
                            embedding_model=args.embedding_model,
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
                     lr=args.lr,
                     weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                gamma=0.5,
                                                step_size=50)
    model.to(device=device)

    if args.pre_chkpnt:
        logger.info(f"Loading the pre-trained model from {args.pre_chkpnt}")
        state_dict = torch.load(args.pre_chkpnt, map_location=device)['model']
        model.load_state_dict(state_dict, strict=False)

    logger.info("Training:")

    log = train(
        model=model,
        accum_iter=args.gard_acc,
        max_num_epochs=args.n_epochs,
        check_val_every=1,
        training_loader=train_dl,
        val_loader=val_dl,
        loss_fun=loss_function,
        lammy=args.lammy,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        patience=20,
        monitor=args.monitor,
        checkpoint_address=args.chkpnt)

    np.save(args.log, log)


if __name__ == "__main__":
    setup_train()