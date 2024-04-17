import torch
import os
from loguru import logger
from Bio import SeqIO
from argparse import ArgumentParser
from transformers import EsmForProteinFolding


def read_fasta(fast_file: str):
    
    data = []

    with open(fast_file) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            data.append({
                'id': str(record.id),
                'seq': str(record.seq)
            })

    return data


def esm2fold(my_data: list,
             result_file: str,
             device: torch.device = torch.device('cpu'),
             batch_size: int = 32):

    esmfold = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    esmfold.to(device)
    esmfold.eval()

    ind = 0
    n_seqs = len(my_data)
    pdbs = []

    while ind < n_seqs:

        logger.info(f'index: {ind}')
        seqs = []

        if ind + batch_size > n_seqs:
            batch_size = n_seqs - ind

        for seq_inf in my_data[ind:ind + batch_size]:
            seqs.append(seq_inf['seq'])

        with torch.no_grad():
            pdbs += esmfold.infer_pdbs(seqs)

        ind += batch_size

    for i in range(len(pdbs)):
        with open(os.path.join(result_file, f"{my_data[i]['id']}.pdb"), 'w') as f:
            f.write(pdbs[i])
            
    return


if __name__ == "__main__":

    parser = ArgumentParser(description='ESM2 folding')
    # data
    parser.add_argument('-seqs', type=str, required=True, 
                        help='.fasta file of sequences')

    parser.add_argument('-out', type=str, required=True,
                        help='pdbs saving directory')

    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    data = read_fasta(args.seqs)

    esm2fold(
        my_data=data,
        result_file=args.out,
        device=device
    )