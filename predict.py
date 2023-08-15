import torch
import numpy as np
import pandas as pd
import torch_geometric.data
from scipy.stats import mode
from loguru import logger
from torch_geometric.loader import DataLoader
from torch.nn import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    Sigmoid
)
import torch.nn.functional as F
from dataset import ToxicityData
from utils import check_loss, cal_metrics, ensemble_structures
import wandb


def ensemble(model,
             checkpoint_folder: str,
             config: dict,
             device: torch.device,
             fasta_seqs: list,
             test_pdbs: str,
             embed_dir: str,
             mode_pred: str,
             threshold: float) -> dict:

    test_data = ToxicityData(seqs_file=fasta_seqs,
                             pdbs_path=test_pdbs,
                             embeddings_dir=embed_dir,
                             device=device,
                             max_d=config['max_distance'])

    logger.info("Loading test datasets")

    test_dl = DataLoader(
        dataset=test_data,
        batch_size=config["batch_size"],
        num_workers=8,
        shuffle=False)

    logger.info("Calculating metrics on test set")

    model.to(device=device)

    if mode_pred == 'train' or mode_pred == 'test':

        model.load_state_dict(torch.load(checkpoint_folder, map_location=device)['model'])

        _, _, preds = ensemble_structures(model=model,
                                          loader=test_dl,
                                          embeddings=test_data.seq_embeddings,
                                          device=device,
                                          threshold=threshold)

        meters = cal_metrics(preds, {})
    else:

        model.load_state_dict(torch.load(checkpoint_folder, map_location=device)['model'])

        ids, weights, preds = ensemble_structures(model=model,
                                                  loader=test_dl,
                                                  embeddings=test_data.seq_embeddings,
                                                  device=device,
                                                  threshold=threshold)
        meters = cal_metrics(preds, {})

        preds['id'] = ids
        df = pd.DataFrame(preds)
        df.to_csv('best_scores.csv', index=False)

        np.savez('weights.npz', **weights)

    return meters
