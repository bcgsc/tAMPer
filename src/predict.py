import torch
import os
import numpy as np
import pandas as pd
from loguru import logger
from torch_geometric.loader import DataLoader
from tAMPer import tAMPer
from collections import OrderedDict
from argparse import ArgumentParser
import torch.nn.functional as F
from dataset import ToxicityData
from utils import set_seed, cal_metrics


def ensemble_structures(model: torch.nn.Module,
                        loader: DataLoader,
                        device: torch.device,
                        threshold: float = 0.5):
    model.eval()
    sigmoid = torch.nn.Sigmoid()

    y_hat = OrderedDict()
    scores = OrderedDict()
    sequences = {}

    for _, graphs in enumerate(loader):
        with torch.no_grad():

            graphs = graphs.to(device)
            embeddings = graphs.embeddings

            preds = model(embeddings, graphs)

            IDs = list(graphs.id)
            seqs = list(graphs.seq)

            for i in range(len(IDs)):
                if IDs[i] in scores.keys():
                    scores[IDs[i]].append(sigmoid(preds['tx'][i]).item())
                else:
                    scores[IDs[i]] = [sigmoid(preds['tx'][i]).item()]
                sequences[IDs[i]] = seqs[i]

    for key in scores.keys():
        scores[key] = np.mean(scores[key])
        y_hat[key] = 1.0 if scores[key] > threshold else 0.0

    predictions = {
        'id': list(scores.keys()),
        'sequence': list(sequences.values()),
        'score': list(scores.values()),
        'prediction': list(y_hat.values())
    }

    return predictions


def predict(model: torch.nn.Module,
            checkpoint_folder: str,
            data_loader: DataLoader,
            device: torch.device,
            threshold: float,
            result_csv: str):

    logger.info("Predicting ...")

    model.to(device=device)
    model.load_state_dict(torch.load(checkpoint_folder, map_location=device)['model'])

    predictions = ensemble_structures(model=model,
                                      loader=data_loader,
                                      device=device,
                                      threshold=threshold)

    out = pd.DataFrame.from_dict(predictions)
    out.to_csv(result_csv, index=False)
    logger.info(f"Saved the predictions at {result_csv}")


if __name__ == "__main__":

    parser = ArgumentParser(description='predict.py script runs tAMPer for prediction.')

    parser.add_argument('-seqs', default=f'{os.getcwd()}/data/sequences/seqs.faa', type=str,
                        required=True, help='sequences fasta file for prediction (.fasta)')

    parser.add_argument('-pdbs', default=f'{os.getcwd()}/data/structures/', type=str,
                        required=True, help='address directory of train structures')

    parser.add_argument('-hdim', default=32, type=int, required=False,
                        help='hidden dimension of model for h_seq and h_strct')

    parser.add_argument('-embedding_model', default="t12", type=str, required=False,
                        help='different variant of ESM2 embeddings: {t6, t12}')

    parser.add_argument('-d_max', default=10, type=int, required=False,
                        help='max distance to consider two connect two residues in the graph')

    parser.add_argument('-chkpnt', default=f'{os.getcwd()}/checkpoints/trained/chkpnt.pt',
                        type=str, required=False, help='address of .pt checkpoint to load the model')

    parser.add_argument('-result_csv', default=f'{os.getcwd()}/results/predictions.csv', type=str,
                        required=False, help='address of results (.csv) to be saved')

    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(1)

    if args.embedding_model == 't6':
        seq_inp_dim = 320
    elif args.embedding_model == 't12':
        seq_inp_dim = 480
    elif args.embedding_model == 't30':
        seq_inp_dim = 640
    else:
        seq_inp_dim = 1280

    data = ToxicityData(seqs=args.seqs,
                        pdbs_path=args.pdbs,
                        embedding_model=args.embedding_model,
                        max_d=args.d_max)

    logger.info("Loading the dataset")

    test_dl = DataLoader(
        dataset=data,
        batch_size=32,
        num_workers=8,
        shuffle=False)

    tamper = tAMPer(
        input_modality='all',
        seq_input_dim=seq_inp_dim,
        node_dims=(6, 3),
        edge_dims=(32, 1),
        node_hdim=(args.hdim, 16),
        edge_hdim=(32, 1),
        n_heads=8,
        gru_hdim=args.hdim,
        n_grus=1,
        n_gnns=1)

    predict(
        model=tamper,
        checkpoint_folder=args.chkpnt,
        data_loader=test_dl,
        device=device,
        result_csv=args.result_csv,
        threshold=0.5
    )
