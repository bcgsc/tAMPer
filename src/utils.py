import torch
import numpy as np
import random
from torch_geometric.nn.pool import global_mean_pool
from scipy.stats import mode
from collections import OrderedDict
import torch_geometric.data
from loguru import logger
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from sklearn import metrics
from torch.nn import Sigmoid
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    return


def check_loss(model: torch.nn.Module,
               loader: DataLoader,
               device: torch.device,
               loss_func: dict,
               lammy: float = 0.1,
               threshold: float = 0.5):

    model.eval()
    loss_hist, ss_hist, tx_hist = list(), list(), list()
    sigmoid = Sigmoid()

    tx_outs = {'y': [],
               'y_hat': [],
               'score': []}

    for _, graphs in enumerate(loader):
        with torch.no_grad():
            graphs = graphs.to(device)
            embeddings = graphs.embeddings

            preds = model(embeddings, graphs)

            tx_loss = loss_func['tx'](preds['tx'].flatten(), graphs.y)
            ss_loss = loss_func['ss'](preds['ss'], graphs.ss.flatten())

            loss = (1 - lammy) * tx_loss + lammy * ss_loss

            scores = sigmoid(preds['tx'].flatten()).cpu()
            y_hat = (scores > threshold).to(torch.float32)

            tx_outs['y'] += graphs.y.cpu().tolist()
            tx_outs['y_hat'] += y_hat.tolist()
            tx_outs['score'] += scores.tolist()

            ss_hist.append(ss_loss.item())
            tx_hist.append(tx_loss.item())
            loss_hist.append(loss.item())

    mean_loss = np.mean(loss_hist)
    mean_ss = np.mean(ss_hist)
    mean_tx = np.mean(tx_hist)

    losses = {'loss': mean_loss,
              'tx_loss': mean_tx,
              'ss_loss': mean_ss}

    logger.info(f"loss: {mean_loss}")
    logger.info(f"tx_loss: {mean_tx}")
    logger.info(f"ss_loss: {mean_ss}")

    return tx_outs, losses


def cal_metrics(y_preds: dict, meters: dict):
    y, y_hat, score = y_preds['y'], y_preds['y_hat'], y_preds['score']

    acc = metrics.accuracy_score(y, y_hat)
    sen = metrics.recall_score(y, y_hat)
    pre = metrics.precision_score(y, y_hat)
    f1 = metrics.f1_score(y, y_hat)
    mcc = metrics.matthews_corrcoef(y, y_hat)
    tn, fp, _, _ = metrics.confusion_matrix(y, y_hat, labels=[0.0, 1.0]).ravel()
    specificity = float(tn / (tn + fp))
    auROC = metrics.roc_auc_score(y, score)
    precision, recall, _ = metrics.precision_recall_curve(y, score)
    auPRC = metrics.auc(recall, precision)

    meters['accuracy'] = acc
    meters['sensitivity'] = sen
    meters['specificity'] = specificity
    meters['precision'] = pre
    meters['f1'] = f1
    meters['mcc'] = mcc
    meters['auROC'] = auROC
    meters['auPRC'] = auPRC

    logger.info(f"sensitivity: {sen}")
    logger.info(f"specificity: {specificity}")
    logger.info(f"f1: {f1}")
    logger.info(f"mcc: {mcc}")
    logger.info(f"auROC: {auROC}")
    logger.info(f"auPRC: {auPRC}")
    logger.info("*****************")

    return meters


def monitor_metric(log_dict: dict,
                   cur_model: torch.nn.Module,
                   chkpnt_addr: str,
                   patience: int,
                   metric: str):

    cur_epoch = len(log_dict["val_metrics"]) - 1

    if 'loss' in metric:
        min_loss, min_epoch = np.inf, 0

        for i in range(len(log_dict['val_metrics'])):
            if log_dict["val_metrics"][i][metric] <= min_loss:
                min_epoch = i
                min_loss = log_dict["val_metrics"][i][metric]

        if cur_epoch == min_epoch:
            logger.info(f"New best {metric} score: {min_loss}")
            trigger_times = 0
        else:
            trigger_times = cur_epoch - min_epoch
    else:
        max_meter, max_epoch = 0.0, 0

        for i in range(len(log_dict['val_metrics'])):
            if log_dict["val_metrics"][i][metric] >= max_meter:
                max_epoch = i
                max_meter = log_dict["val_metrics"][i][metric]

        if cur_epoch == max_epoch:
            logger.info(f"New best {metric} score: {max_meter}")
            trigger_times = 0
        else:
            trigger_times = cur_epoch - max_epoch

    if trigger_times == 0:
        logger.info(f"Saving model to checkpoint_address")
        torch.save({'model': cur_model.state_dict()}, chkpnt_addr)

    if trigger_times >= patience:
        return True
    else:
        return False


def merge(pos_fasta: str, neg_fasta: str) -> list:
    data = []

    with open(pos_fasta) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            data.append({'id': str(record.id),
                         'seq': str(record.seq),
                         'AMD': 1 if '_AMD' in record.id else 0,
                         'label': 1})

    with open(neg_fasta) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            data.append({'id': str(record.id),
                         'seq': str(record.seq),
                         'AMD': 1 if '_AMD' in record.id else 0,
                         'label': 0})
    return data


def read_fasta(fasta_file: str, label: int = -1) -> list:
    data = []
    with open(fasta_file) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            data.append({'id': str(record.id),
                         'seq': str(record.seq),
                         'AMD': 1 if '_AMD' in record.id else 0,
                         'label': label})
    return data


def retrieve_best_model(log_dict: dict, metric: str):
    best_epoch = 0

    if metric == 'loss':
        min_loss, = np.inf
        for i in range(len(log_dict['val_metrics'])):
            if log_dict["val_metrics"][i][metric] <= min_loss:
                best_epoch = i
                min_loss = log_dict["val_metrics"][i][metric]
    else:
        max_meter = 0.0
        for i in range(len(log_dict['val_metrics'])):
            if log_dict["val_metrics"][i][metric] >= max_meter:
                best_epoch = i
                max_meter = log_dict["val_metrics"][i][metric]

    return log_dict["val_metrics"][best_epoch]


def calculate_pos_weight(n_pos: int, n_neg: int, beta: float):
    samples_per_class = torch.tensor([n_neg, n_pos])
    effective_num = 1.0 - torch.pow(beta, samples_per_class)

    weights = (1.0 - beta) / effective_num
    weights = weights / (torch.sum(weights) * 2)

    pos_weight = weights[1] / weights[0]

    return pos_weight


def pre_metrics(model: torch.nn.Module,
                loader: DataLoader,
                device: torch.device,
                loss_func: torch.nn.Module, ):
    model.eval()
    loss_hist = list()

    num_corrects = 0
    num_samples = 0

    for _, graphs in enumerate(loader):
        with torch.no_grad():
            graphs = graphs.to(device)

            preds = model(graphs)
            target = graphs.aa.flatten()
            loss = loss_func(preds, target)

            scores = preds.cpu()
            y_hat = torch.argmax(scores, dim=1)
            corrects = (y_hat == target.cpu()).to(torch.float32)
            seqs_acc = global_mean_pool(corrects.unsqueeze(1),
                                        batch=graphs.batch.cpu())

            num_corrects += seqs_acc.squeeze().sum().item()
            num_samples += graphs.batch.max().cpu().item() + 1

        loss_hist.append(loss.item())

    mean_loss = np.mean(loss_hist)
    acc = float(num_corrects / num_samples)

    meters = {'loss': mean_loss,
              'acc': acc}

    logger.info(f"loss: {mean_loss}")
    logger.info(f"accuracy: {acc}")

    return meters
