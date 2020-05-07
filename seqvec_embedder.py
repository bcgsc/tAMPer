"""
Date: 2020-02-27
Author: figalit (github.com/figalit)

Adapted mostly from the seqvec_embedder.py from the SeqVec Github page.
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from allennlp.commands.elmo import ElmoEmbedder

MAX_CHARS = 15000
EMB_LEN = 1024
MODEL_DIR = '/projects/btl/ftaho/toxic_amps/seqvec/uniref50_v2/'
CPU = False
TIMESTEPS = 100

def get_elmo_model(seqvec_dir=MODEL_DIR):
    model_dir = Path(seqvec_dir)
    weights_path = model_dir / 'weights.hdf5'
    options_path = model_dir / 'options.json'

    # if no pre-trained model is available, yet --> download it
    if not (weights_path.exists() and options_path.exists()):
        print('No existing model found. Start downloading pre-trained SeqVec (~360MB)...')
        import urllib.request
        Path.mkdir(seqvec_dir)
        repo_link    = 'http://rostlab.org/~deepppi/embedding_repo/embedding_models/seqvec'
        options_link = repo_link +'/options.json'
        weights_link = repo_link +'/weights.hdf5'
        urllib.request.urlretrieve( options_link, options_path )
        urllib.request.urlretrieve( weights_link, weights_path )

    cuda_device = 0 if torch.cuda.is_available() and not CPU else -1
    return ElmoEmbedder( weight_file=weights_path, options_file=options_path, cuda_device=cuda_device)

def process_embedding(embedding):
    embedding = torch.tensor(embedding)
    embedding = embedding.sum(dim=0)
    embedding = embedding.mean(dim=0) # for whole protein
    return embedding.cpu().detach().numpy()

def process_embedding_per_residue(embedding):
    embedding = torch.tensor(embedding)
    embedding = embedding.sum(dim=0)
    return embedding.cpu().detach().numpy()

def get_sequences(csvfile):
    """
    Given a CSV file of sequence,label, return a dict of id to sequence, where id is assigned acc. to order."
    """
    sequences = dict()

    df = pd.read_csv(csvfile)
    ids = list(range(len(df)))
    df['id'] = ids
    
    for elem in df.iterrows():
        seq = elem[1][0]
        seqid = elem[1][2]
        sequences[seqid] = seq
    return sequences, df

def get_embeddings(seq_dict, df, out_path, per_residue=False):
    emb_dict = dict()
    # For a time speed up, sort the sequences based on length.
    seq_dict = sorted(seq_dict.items(), key=lambda kv: len( seq_dict[kv[0]]))
    
    model = get_elmo_model()
    batch = list()
    length_counter = 0
    for index, (identifier, sequence) in enumerate(seq_dict): # for all sequences in the set
        batch.append((identifier, sequence))
        length_counter += len(sequence)

        if length_counter > MAX_CHARS or len(sequence)>MAX_CHARS/2 or index==len(seq_dict)-1:
            tokens = [ list(seq) for _, seq in batch]
            embeddings = model.embed_sentences(tokens)

            runtime_error = False
            for batch_idx, (sample_id, seq) in enumerate(batch): # for each seq in the batch
                try: embedding = next(embeddings)
                except RuntimeError:
                    print('RuntimeError for {} (len={}).'.format(sample_id,len(seq)))
                    print('Starting single sequence processing')
                    break
                if per_residue:
                    emb_dict[str(sample_id)] = process_embedding_per_residue(embedding)
                else:
                    emb_dict[sample_id] = process_embedding(embedding)
            # This should not happen! There will be a keyerror later if it does.
            if runtime_error:
                for batch_idx, (sample_id, seq) in enumerate(batch):
                    try: embedding = model.embed_sentence(tokens[batch_idx])
                    except RuntimeError:
                        print('RuntimeError for {} (len={}).'.format(sample_id,len(seq)))
                        print('Single sequence processing not possible. Skipping seq. ..' + 
                              'Consider splitting the sequence into smaller seqs or process on CPU.')
                        continue
                    if per_residue:
                        emb_dict[str(sample_id)] = process_embedding_per_residue(embedding)
                    else:
                        emb_dict[sample_id] = process_embedding(embedding)

            batch = list()
            length_counter = 0
    print('\nTotal number of embeddings: {}'.format(len(emb_dict)))

    # Write out files for the sequences, the labels so that further analysis can continue smoothly.
    assert(len(emb_dict) == len(df))
    
    if per_residue:
        # save the embeddings as npz file
        np.savez( "{}_per_residue".format(out_path), **emb_dict)
        labelfile = open("{}_labels_per_residue.csv".format(out_path), 'w+')
        labelfile.write("label,seqid\n")
        for i, elem in enumerate(df.iterrows()):
            seq =  elem[1][0]
            label = elem[1][1]
            seqid = elem[1][2]
            labelfile.write("{},{}\n".format(label,seqid))
        labelfile.close()
        return None,None

    X_numpy = np.zeros((len(emb_dict), EMB_LEN))
    y_numpy = np.zeros((len(emb_dict),))
    for i, elem in enumerate(df.iterrows()):
        seq =  elem[1][0]
        label = elem[1][1]
        seqid = elem[1][2]
        X_numpy[i,:] = emb_dict[seqid]#[0]
        if label == 'toxin': y_numpy[i] = 1

    pd.DataFrame(X_numpy).to_csv("{}.csv".format(out_path), header=None, index=None)
    pd.DataFrame(y_numpy).to_csv("{}_labels.csv".format(out_path), header=None, index=None)
    return X_numpy, y_numpy

def create_arg_parser():
    parser = argparse.ArgumentParser(description=('embedder.py creates ELMo embeddings for a file containing sequence,label columns in CSV format.'))
    parser.add_argument( '-i', '--input', required=True, type=str, help='A path to a csv-formatted text file containing sequence,label annotation(s).')
    parser.add_argument( '-o', '--output', required=True, type=str, help='A path to a file for saving the created embeddings as csv file.')
    return parser

def get_single_seq_embedding(sequence):
    """Utility for single amino acid sequence embedding."""
    model = get_elmo_model()
    tokens = list(sequence)
    embedding = model.embed_sentence(tokens)
    return process_embedding(embedding)

# TODO: Delete.
def get_list_embedding(sequence_list):
    """Utility for a list of amino acid sequence embeddings."""
    model = get_elmo_model()
    result = np.zeros((len(sequence_list), EMB_LEN))
    for i,sequence in enumerate(sequence_list):
        tokens = list(sequence)
        embedding = model.embed_sentence(tokens)
        result[i,:] = process_embedding(embedding)
    return result

def list_embeddings(sequence_list, seqvec_dir):
    """Utility for a list of amino acid sequence embeddings."""
    model = get_elmo_model(seqvec_dir=seqvec_dir)
    result = np.zeros((len(sequence_list), EMB_LEN))
    for i,sequence in enumerate(sequence_list):
        tokens = list(sequence)
        embedding = model.embed_sentence(tokens)
        result[i,:] = process_embedding(embedding)
    return result

def list_embeddings_residue(sequence_list, seqvec_dir):
    """Utility for a list of amino acid sequence embeddings to get per residue embeddings."""
    model = get_elmo_model(seqvec_dir=seqvec_dir)

    result = np.zeros((len(seqlist), TIMESTEPS, EMB_LEN))
    for i,sequence in enumerate(sequence_list):
        tokens = list(sequence)
        embedding = model.embed_sentence(tokens)
        mat = process_embedding_per_residue(embedding)
        l = mat.shape[0]
        result[i,:l,:] = mat 
    # print(result.shape)
    return result


# TODO: Delete.
def get_list_embedding_per_residue(sequence_list):
    """Utility for a list of amino acid sequence embeddings to get per residue embeddings."""
    model = get_elmo_model()
    # First ensure only sequences with length <= 100 are kept.
    seqlist = []
    for seq in sequence_list:
        if len(seq) > 100: continue
        seqlist.append(seq)
    print("{} sequences were removed due to length > 100.".format(len(sequence_list) - len(seqlist)))

    timesteps = 100
    n = len(seqlist)
    result = np.zeros((n, timesteps, EMB_LEN))
    for i,sequence in enumerate(sequence_list):
        tokens = list(sequence)
        embedding = model.embed_sentence(tokens)
        mat = process_embedding_per_residue(embedding)
        l = mat.shape[0]
        result[i,:l,:] = mat 
    print(result.shape)
    return result

def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    seq_dir   = Path(args.input)
    emb_path  = Path(args.output)

    seq_dict, df = get_sequences(seq_dir)
    X, _ = get_embeddings(seq_dict, df, emb_path)
    print(X.shape) #, y.shape)

if __name__ == '__main__':
    main()