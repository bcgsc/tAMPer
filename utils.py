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
               beta: float = 0.0,
               threshold: float = 0.5,
               embeddings: dict = None,
               pretrain: bool = False):

    model.eval()
    loss_hist, ss_hist, tx_hist = list(), list(), list()
    sigmoid = Sigmoid()

    tx_outs = {'y': [],
               'y_hat': [],
               'score': []}

    num_corrects = 0
    num_samples = 0

    for _, graphs in enumerate(loader):
        with torch.no_grad():

            if not pretrain:
                graphs = graphs.to(device)

                sequences = [embeddings[id] for id in graphs.id]
                sequences = torch.cat(sequences).to(device)

                preds = model(sequences, graphs)

                tx_loss = loss_func['tx'](preds['tx'].flatten(), graphs.y)
                ss_loss = loss_func['ss'](preds['ss'], graphs.ss.flatten())

                loss = (1 - beta) * tx_loss + beta * ss_loss

                scores = sigmoid(preds['tx'].flatten()).cpu()
                y_hat = (scores > threshold).to(torch.float32)

                num_corrects += (y_hat == graphs.y.cpu()).sum().item()
                num_samples += graphs.y.shape[0]

                tx_outs['y'] += graphs.y.cpu().tolist()
                tx_outs['y_hat'] += y_hat.tolist()
                tx_outs['score'] += scores.tolist()

                ss_hist.append(ss_loss.item())
                tx_hist.append(tx_loss.item())
            else:
                graphs = graphs.to(device)

                preds = model(graphs)
                target = graphs.aa.flatten()
                loss = loss_func['aa'](preds['aa'], target)

                scores = preds['aa'].cpu()
                y_hat = torch.argmax(scores, dim=1)
                corrects = (y_hat == target.cpu()).to(torch.float32)
                seqs_acc = global_mean_pool(corrects.unsqueeze(1),
                                            batch=graphs.batch.cpu())

                num_corrects += seqs_acc.squeeze().sum().item()
                num_samples += graphs.batch.max().cpu().item() + 1

            loss_hist.append(loss.item())

    mean_loss = np.mean(loss_hist)
    mean_ss, mean_tx, acc = 0.0, 0.0, 0.0

    if ss_hist:
        mean_ss = np.mean(ss_hist)
        mean_tx = np.mean(tx_hist)

    if num_samples != 0:
        acc = float(num_corrects / num_samples)

    meters = {'loss': mean_loss,
              'tx_loss': mean_tx,
              'ss_loss': mean_ss,
              'accuracy': acc}

    logger.info(f"loss: {mean_loss}")
    logger.info(f"tx_loss: {mean_tx}")
    logger.info(f"ss_loss: {mean_ss}")
    logger.info(f"accuracy: {acc}")

    return tx_outs, meters


def cal_metrics(y_preds: dict, meters: dict):
    y, y_hat, score = y_preds['y'], y_preds['y_hat'], y_preds['score']

    acc = metrics.accuracy_score(y, y_hat)
    sen = metrics.recall_score(y, y_hat)
    pre = metrics.precision_score(y, y_hat)
    f1 = metrics.f1_score(y, y_hat)
    mcc = metrics.matthews_corrcoef(y, y_hat)
    tn, fp, fn, tp = metrics.confusion_matrix(y, y_hat, labels=[0.0, 1.0]).ravel()
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


# only for predicting
def ensemble_structures(model: torch.nn.Module, loader: DataLoader, embeddings: dict,
                        device: torch.device, threshold: float = 0.5):
    model.eval()

    y_hat = OrderedDict()
    scores = OrderedDict()
    y_true = OrderedDict()
    weights = OrderedDict()

    sigmoid = Sigmoid()

    for _, graphs in enumerate(loader):
        with torch.no_grad():

            sequences = [embeddings[id] for id in graphs.id]

            sequences = torch.cat(sequences).to(device)
            graphs = graphs.to(device)

            preds = model(sequences, graphs)
            weight = preds['weights']

            IDs = list(graphs.id)

            for i in range(len(IDs)):
                if IDs[i] in scores.keys():
                    scores[IDs[i]].append(sigmoid(preds['tx'][i]).item())
                    weights[f'{IDs[i]}_{len(scores[IDs[i]])-1}'] = [weight[i].cpu().numpy()]
                else:
                    scores[IDs[i]] = [sigmoid(preds['tx'][i]).item()]
                    weights[f'{IDs[i]}_0'] = [weight[i].cpu().numpy()]

                y_true[IDs[i]] = graphs.y[i].item()

    for key in scores.keys():
        scores[key] = np.mean(scores[key])
        y_hat[key] = 1.0 if scores[key] > threshold else 0.0

    ids = list(scores.keys())

    y_predictions = {
        'score': np.array(list(scores.values())),
        'y_hat': np.array(list(y_hat.values())),
        'y': np.array(list(y_true.values()))
    }

    return ids, weights, y_predictions


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


def merge(fasta_seqs: list) -> list:
    pos_fasta, neg_fasta = fasta_seqs
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


def read_fasta(fasta_file: str) -> list:
    data = []
    with open(fasta_file) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            data.append({'id': str(record.id),
                         'seq': str(record.seq),
                         'AMD': 1 if '_AMD' in record.id else 0,
                         'label': -1})
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