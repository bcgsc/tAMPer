import torch
import os
import numpy as np
import pandas as pd
from loguru import logger
from torch_geometric.loader import DataLoader
from tAMPer import tAMPer
from argparse import ArgumentParser
import torch.nn.functional as F
from dataset import ToxicityData, SimpleData
from utils import set_seed, ensemble_structures, cal_metrics


def predict(model,
            checkpoint_folder: str,
            data_loader: DataLoader,
            device: torch.device,
            threshold: float,
            result_csv: str):
    logger.info("Calculating metrics on test set")

    model.to(device=device)
    model.load_state_dict(torch.load(checkpoint_folder, map_location=device)['model'])

    ids, preds = ensemble_structures(model=model,
                                     loader=data_loader,
                                     device=device,
                                     threshold=threshold)

    # metrics = cal_metrics(preds, {})
    # df = pd.DataFrame({key: [val] for key, val in metrics.items()})
    # df.to_csv(result_csv, index=False)

    preds['id'] = ids

    out = pd.DataFrame.from_dict(preds)
    out.to_csv(result_csv, index=False)


if __name__ == "__main__":

    parser = ArgumentParser(description='predict.py script runs tAMPer for prediction.')

    parser.add_argument('-pos', default=f'{os.getcwd()}/data/sequences/tr_pos.faa', type=str,
                        required=True, help='training toxic sequences fasta file (.fasta)')

    parser.add_argument('-neg', default=f'{os.getcwd()}/data/sequences/tr_neg.faa', type=str,
                        required=True, help='training non-toxic sequences fasta file (.fasta)')

    parser.add_argument('-pdb', default=f'{os.getcwd()}/data/structures/', type=str,
                        required=True, help='address directory of train structures')

    parser.add_argument('-embed', default=f'{os.getcwd()}/data/embeddings/', type=str,
                        required=True, help='address directory of train embeddings')

    parser.add_argument('-hdim', default=32, type=int, required=False,
                        help='hidden dimension of model for h_seq and h_strct')

    parser.add_argument('-embedding_model', default="t6", type=str, required=False,
                        help='different variant of ESM2 embeddings: {t6, t12}')

    parser.add_argument('-dm', default=10, type=int, required=False,
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

    test_data = SimpleData(pos_seqs=args.pos,
                           neg_seqs=args.neg,
                           graphs_dir=args.pdb,
                           embeddings_dir=args.embed)

    logger.info("Loading test datasets")

    test_dl = DataLoader(
        dataset=test_data,
        batch_size=32,
        num_workers=8,
        shuffle=False)

    tamper = tAMPer(
        input_modality='all',
        seq_input_dim=seq_inp_dim,
        node_dims=(9, 3),
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