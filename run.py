from argparse import ArgumentParser
import pandas as pd
from train import setup_train
from tAMPer import tAMPer
from pretraining.model import GraphNN
from CV import cross_validation
from pretraining.pre_train import setup_pretrain
from utils import set_seed, retrieve_best_model
from predict import ensemble
import torch
from loguru import logger
import os


def main():
    parser = ArgumentParser()
    parser.add_argument('-env', '--environment', default='local', type=str, required=True)
    parser.add_argument('-tr_pos', '--tr_pos', default=f'{os.getcwd()}/data/sequences/tr_pos.faa', type=str, required=True)
    parser.add_argument('-tr_neg', '--tr_neg', default=f'{os.getcwd()}/data/sequences/tr_neg.faa', type=str, required=True)
    parser.add_argument('-val_pos', '--val_pos', default=f'{os.getcwd()}/data/sequences/val_pos.faa', type=str, required=False)
    parser.add_argument('-val_neg', '--val_neg', default=f'{os.getcwd()}/data/sequences/val_neg.faa', type=str, required=False)
    parser.add_argument('-pdb_dir', '--pdb_dir', default=f'{os.getcwd()}/data/structures/', type=str, required=True)
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
    parser.add_argument('-mode', '--mode', default="train", type=str, required=False)
    parser.add_argument('-mdl', '--modality', default="all", type=str, required=False)
    parser.add_argument('-dm', '--d_max', default=10, type=int, required=False)
    parser.add_argument('-beta', '--beta', default=0.0, type=float, required=False)
    parser.add_argument('-monitor', '--monitor_meter', default='loss', type=str, required=False)
    parser.add_argument('-pdata', '--pre_data', default='AF2', type=str, required=False)
    parser.add_argument('-ck', '--checkpoint_dir', default=f'{os.getcwd()}/checkpoints/chkpnt.pt', type=str, required=False)
    parser.add_argument('-pck', '--pre_chkpnt', default=f'{os.getcwd()}/checkpoints/trained/pre_GNNs.pt', type=str, required=False)
    parser.add_argument('-log', '--log_file', default=f'{os.getcwd()}/logs/log.npy', type=str, required=False)
    parser.add_argument('-res', '--res', default=f'{os.getcwd()}/results/val.csv', type=str, required=False)
    parser.add_argument('-name', '--name', default="", type=str, required=False)
    parser.add_argument('-step', '--step', default=50, type=int, required=False)
    parser.add_argument('-seed', '--seed', default=1, type=int, required=False)
    parser.add_argument('-dataset', '--dataset', default='hemolysis', type=str, required=False)
    parser.add_argument('-wdb', '--wandb', action='store_true')

    args = parser.parse_args()

    set_seed(args.seed)

    if args.environment == 'gsc':

        if args.dataset == 'toxDL':  # toxDL
            train_positives = "/projects/amp/tAMPer/benchmark/toxDL_data/data/train/sequences/train_toxic.faa"
            train_negatives = "/projects/amp/tAMPer/benchmark/toxDL_data/data/train/sequences/train_NotToxic.faa"

            val_positives = "/projects/amp/tAMPer/benchmark/toxDL_data/data/val/sequences/val_toxic.faa"
            val_negatives = "/projects/amp/tAMPer/benchmark/toxDL_data/data/val/sequences/val_NotToxic.faa"

            pdbs_dir = "/projects/amp/tAMPer/benchmark/toxDL_data/data/structures/"
            embed_dir = f"/projects/amp/tAMPer/benchmark/toxDL_data/data/embeddings/{args.embedding}/"

            test_positives = "/projects/amp/tAMPer/benchmark/toxDL_data/data/test/sequences/test_toxic.faa"
            test_negatives = "/projects/amp/tAMPer/benchmark/toxDL_data/data/test/sequences/test_NotToxic.faa"

            test_pdbs = pdbs_dir
            test_embed = embed_dir
        else:
            train_positives = "/projects/amp/tAMPer/manuscript/data/hemolysis/sequences/tr_hemolytic.faa"
            train_negatives = "/projects/amp/tAMPer/manuscript/data/hemolysis/sequences/tr_non_hemolytic.faa"

            val_positives = "/projects/amp/tAMPer/manuscript/data/hemolysis/sequences/val_hemolytic.faa"
            val_negatives = "/projects/amp/tAMPer/manuscript/data/hemolysis/sequences/val_non_hemolytic.faa"

            pdbs_dir = "/projects/amp/tAMPer/manuscript/data/hemolysis/structures/"
            embed_dir = f"/projects/amp/tAMPer/manuscript/data/hemolysis/embeddings/{args.embedding}/"

            test_positives = "/projects/amp/tAMPer/colabfold_pred/tAMPer_test/test_sequences/toxic_test.faa"
            test_negatives = "/projects/amp/tAMPer/colabfold_pred/tAMPer_test/test_sequences/non_toxic_test.faa"

            test_pdbs = "/projects/amp/tAMPer/colabfold_pred/tAMPer_test/test_structures/"
            test_embed = f"/projects/amp/tAMPer/colabfold_pred/tAMPer_test/test_embeddings/{args.embedding}/"

        pre_seqs = '/projects/amp/tAMPer/manuscript/data/pre_data/sequences/new_seqs.faa'
        pre_strcts = '/projects/amp/tAMPer/manuscript/data/pre_data/structures/'

        if args.modality == 'structure':
            embed_dir = None
            test_embed = None

    elif args.environment == 'cedar':

        if args.dataset == 'toxDL':  # toxDL

            train_positives = "/home/hebz/projects/def-ibirol/hebz/toxDL/train/sequences/train_toxic.faa"
            train_negatives = "/home/hebz/projects/def-ibirol/hebz/toxDL/train/sequences/train_NotToxic.faa"

            val_positives = "/home/hebz/projects/def-ibirol/hebz/toxDL/val/sequences/val_toxic.faa"
            val_negatives = "/home/hebz/projects/def-ibirol/hebz/toxDL/val/sequences/val_NotToxic.faa"

            pdbs_dir = "/home/hebz/projects/def-ibirol/hebz/toxDL/structures/"
            embed_dir = f"/home/hebz/projects/def-ibirol/hebz/toxDL/embeddings/{args.embedding}/"

            test_positives = "/home/hebz/projects/def-ibirol/hebz/toxDL/test/sequences/test_toxic.faa"
            test_negatives = "/home/hebz/projects/def-ibirol/hebz/toxDL/test/sequences/test_NotToxic.faa"

            test_pdbs = pdbs_dir
            test_embed = embed_dir

        else:
            train_positives = "/home/hebz/projects/def-ibirol/hebz/hemolysis/sequences/tr_hemolytic.faa"
            train_negatives = "/home/hebz/projects/def-ibirol/hebz/hemolysis/sequences/tr_non_hemolytic.faa"

            val_positives = "/home/hebz/projects/def-ibirol/hebz/hemolysis/sequences/val_hemolytic.faa"
            val_negatives = "/home/hebz/projects/def-ibirol/hebz/hemolysis/sequences/val_non_hemolytic.faa"

            pdbs_dir = "/home/hebz/projects/def-ibirol/hebz/hemolysis/structures/"
            embed_dir = f"/home/hebz/projects/def-ibirol/hebz/hemolysis/embeddings/{args.embedding}/"

            test_positives = "/home/hebz/projects/def-ibirol/hebz/hemolysis/test/sequences/toxic_test.faa"
            test_negatives = "/home/hebz/projects/def-ibirol/hebz/hemolysis/test/sequences/non_toxic_test.faa"

            test_pdbs = "/home/hebz/projects/def-ibirol/hebz/hemolysis/test/structures/"
            test_embed = f"/home/hebz/projects/def-ibirol/hebz/hemolysis/test/embeddings/{args.embedding}/"

        pre_seqs = '/home/hebz/projects/def-ibirol/hebz/pre_data/sequences/new_seqs.faa'
        pre_strcts = '/home/hebz/projects/def-ibirol/hebz/pre_data/structures/'

        if args.modality == 'structure':
            embed_dir = None
            test_embed = None
    else:
        train_positives = "/Users/hossein/Desktop/tAMPer/model/check_data/sequences/toxic_sequences.faa"
        train_negatives = "/Users/hossein/Desktop/tAMPer/model/check_data/sequences/not_toxic_sequences.faa"

        val_positives = "/Users/hossein/Desktop/tAMPer/model/check_data/sequences/toxic_sequences.faa"
        val_negatives = "/Users/hossein/Desktop/tAMPer/model/check_data/sequences/not_toxic_sequences.faa"

        pdbs_dir = "/Users/hossein/Desktop/tAMPer/model/check_data/structures/"
        embed_dir = "/Users/hossein/Desktop/model/tAMPer/embed/"

        test_positives = val_positives
        test_negatives = val_negatives
        test_pdbs = pdbs_dir
        test_embed = embed_dir

        if args.modality == 'structure':
            embed_dir = None

        pre_seqs = train_positives
        pre_strcts = pdbs_dir

    hparams = {
        "max_epochs": args.num_epoch,
        "val_every": 1,
        "patience": 50,
        "warmup": args.warmup,
        'douts': {'seq': args.seq_dropout,
                  'strct': args.strct_dropout},
        "monitor": args.monitor_meter,
        "wdb_name": args.name,
        "optimizer_lr": args.learning_rate,
        "step": args.step,
        "beta": args.beta,
        "pre_data": args.pre_data,
        "optimizer_weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "accum_iter": args.accum_iter,
        "embedding_method": args.embedding,
        "max_distance": args.d_max,
        "wandb": args.wandb,
        "gnn_layers": args.gnn_layers,
        "modality": args.modality,
        "env": args.environment}

    gpu_available = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.embedding == 't6':
        seq_inp_dim = 320
    elif args.embedding == 't12':
        seq_inp_dim = 480
    elif args.embedding == 't30':
        seq_inp_dim = 640
    else:
        seq_inp_dim = 1280

    GNN = GraphNN(
        node_dims=(6, 3),
        edge_dims=(32, 1),
        node_h_dim=(args.hdim, 16),
        edge_h_dim=(32, 1),
        num_gnn_layers=args.gnn_layers,
        num_classes=20)

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
    #
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # exit()
    if args.mode == 'pretrain':

        training_log = setup_pretrain(config=hparams,
                                      device=gpu_available,
                                      model=GNN,
                                      tr_seqs=[pre_seqs],
                                      pdbs_dir=pre_strcts,
                                      checkpoint_dir=args.checkpoint_dir,
                                      log_file=args.log_file)
    elif args.mode == 'train':

        training_log = setup_train(config=hparams,
                                   device=gpu_available,
                                   tAMPer=model,
                                   GNN=GNN,
                                   tr_seqs=[train_positives, train_negatives],
                                   val_seqs=[val_positives, val_negatives],
                                   tr_pdbs=pdbs_dir,
                                   val_pdbs=pdbs_dir,
                                   tr_embed=embed_dir,
                                   val_embed=embed_dir,
                                   pre_chkpnt=args.pre_chkpnt,
                                   checkpoint_dir=args.checkpoint_dir,
                                   log_file=args.log_file)

        # meters_val = retrieve_best_model(log_dict=training_log, metric=args.monitor_meter)

        meters_test = ensemble(
            config=hparams,
            device=gpu_available,
            model=model,
            fasta_seqs=[test_positives, test_negatives],
            test_pdbs=test_pdbs,
            embed_dir=test_embed,
            checkpoint_folder=args.checkpoint_dir,
            mode_pred=args.mode,
            threshold=0.5)

        res_list = {
        }

        for key in meters_test.keys():
            res_list[key] = [meters_test[key]]

        if os.path.isfile(args.res):
            csv_file = pd.read_csv(args.res)
            csv_file.loc[len(csv_file.index)] = [item[0] for item in list(res_list.values())]
        else:
            csv_file = pd.DataFrame.from_dict(res_list)

        csv_file.to_csv(args.res, index=False)


if __name__ == "__main__":
    main()
