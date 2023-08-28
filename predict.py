import torch
import os
import numpy as np
import pandas as pd
import torch_geometric.data
from loguru import logger
from torch_geometric.loader import DataLoader
from torch.nn import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    Sigmoid
)
from argparse import ArgumentParser
import torch.nn.functional as F
from dataset import ToxicityData
from utils import check_loss, cal_metrics, ensemble_structures, set_seed


def predict(model,
            checkpoint_folder: str,
            device: torch.device,
            threshold: float) -> dict:

    logger.info("Calculating metrics on test set")

    model.to(device=device)
    model.load_state_dict(torch.load(checkpoint_folder, map_location=device)['model'])

    ids, weights, preds = ensemble_structures(model=model,
                                              loader=test_dl,
                                              embeddings=test_data.seq_embeddings,
                                              device=device,
                                              threshold=threshold)

    preds['id'] = ids
    preds.pop('y')
    preds['prediction'] = preds.pop('y_hat')
    preds['probability'] = preds.pop('score')

    df = pd.DataFrame(preds)
    df.to_csv(args.result_dir, index=False)


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('-seq', '--sequences', type=str, required=True)
    parser.add_argument('-pdb_dir', '--pdb_dir', default=f'{os.getcwd()}/data/structures/', type=str, required=True)
    parser.add_argument('-dm', '--d_max', default=10, type=int, required=False)
    # model configs
    parser.add_argument('-hdim', '--hdim', default=64, type=int, required=False)
    parser.add_argument('-sn', '--sequence_num_layers', default=1, type=int, required=False)
    parser.add_argument('-emd', '--embedding', default="t6", type=str, required=False)
    parser.add_argument('-gl', '--gnn_layers', default=1, type=int, required=False)

    parser.add_argument('-ck', '--checkpoint_dir', default=f'{os.getcwd()}/checkpoints/trained/chkpnt.pt',
                        type=str, required=False)
    parser.add_argument('-res', '--result_dir', default=f'{os.getcwd()}/results/predictions.csv', type=str,
                        required=False)

    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(1)

    if args.embedding == 't6':
        seq_inp_dim = 320
    elif args.embedding == 't12':
        seq_inp_dim = 480
    elif args.embedding == 't30':
        seq_inp_dim = 640
    else:
        seq_inp_dim = 1280

    test_data = ToxicityData(seqs_file=[args.sequences],
                             pdbs_path=args.pdb_dir,
                             device=device,
                             max_d=args.d_max)

    logger.info("Loading test datasets")

    test_dl = DataLoader(
        dataset=test_data,
        batch_size=32,
        num_workers=8,
        shuffle=False)

    tamper = tAMPer(
        seq_input_dim=seq_inp_dim,
        douts={'seq': 0.5, 'strct': 0.5},
        node_dims=(6, 3),
        edge_dims=(32, 1),
        node_h_dim=(args.hdim, 16),
        edge_h_dim=(32, 1),
        gru_hidden_dim=int(args.hdim / 2),  # bi-directional
        gru_layers=args.sequence_num_layers,
        num_gnn_layers=args.gnn_layers)

    predict(
        model=tamper,
        checkpoint_folder=rgs.checkpoint_dir,
        device=device,
        threshold=0.5
    )
