import torch
import os
from loguru import logger
from Bio import SeqIO
from argparse import ArgumentParser
from transformers import EsmForProteinFolding


def read_fasta(fast_file: str):
    data = {
        'id': [],
        'seq': [],
        'pdb': []
    }

    with open(fast_file) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            data['id'].append(str(record.id))
            data['seq'].append(str(record.seq))
    return data


def esm2fold(fasta_file: str,
             result_file: str,
             device: torch.device = torch.device('cpu'),
             batch_size: int = 16):

    my_data = read_fasta(fasta_file)
    esmfold = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    esmfold.to(device)
    esmfold.eval()

    ind = 0
    n_seqs = len(my_data['seq'])

    while ind < n_seqs:

        logger.info(f'index: {ind}')

        if ind + batch_size > n_seqs:
            batch_size = n_seqs - ind

        with torch.no_grad():
            my_data['pdb'] += esmfold.infer_pdbs(my_data['seq'][ind:ind + batch_size])

        ind += batch_size

    for i in range(len(my_data['id'])):
        with open(os.path.join(result_file, f"{my_data['id'][i]}.pdb"), 'w') as f:
            f.write(my_data['pdb'][i])
            
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

    esm2fold(
        fasta_file=args.seqs,
        result_file=args.out,
        device=device
    )