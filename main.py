"""
Date: 2020-02-14
Author: figalit (github.com/figalit)
"""

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import argparse
import sys
import os
import tamper

DEFAULT_SEQVEC_PATH = "seqvec/uniref50_v2" # The default path of seqvec pretrained model, change as necessary.
HELP_TEXT = '''
Use the 'train' mode to input your own positive and negative sequences, and train our models on your sequences.
Use the 'predict' mode to get predictions for your test set.
'''
PREDICTIONS_DIR = "predictions/"
MODELS_DIR = "models/"

def read_fasta(filename):
    from Bio import SeqIO
    output = []
    ids = []
    longer100 = 0
    for seq_record in SeqIO.parse(filename, "fasta"):
        seq = str(seq_record.seq)
        ident = str(seq_record.id)
        if len(seq) > 100: 
            longer100+=1
            continue
        output.append(seq)
        ids.append(ident)
    if longer100 != 0: print('Removed', longer100, 'sequences longer than 100 from', filename)
    return output, ids

def check_seqvec_path(seqvec_path):
    if not os.path.isdir('./{}'.format(seqvec_path)):
        print('Please provide a valid seqvec trained model path (--seqvec_path).'+
            'Consider adding the model under seqvec/uniref50_v2 for simplicity (default value).')
        exit()
    else: print("Using SeqVec model path: {}".format(seqvec_path))

def train(args):
    if args.neg == None or args.pos == None:
        print("Please enter a positive (--pos) and negative (--neg) fasta file.")
        exit()
    
    check_seqvec_path(args.seqvec_path)
    positive_seqs, _ = read_fasta(args.pos)
    negative_seqs, _ = read_fasta(args.neg)

    models_dir = "{}{}".format(MODELS_DIR, args.name)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    tamper.embed(pos_seqs=positive_seqs, neg_seqs=negative_seqs,  models_dir=models_dir, seqvec_path=args.seqvec_path)

def predict(args):
    if args.sequences == None:
        print("Please enter a fasta file of sequences for prediction.")
        exit()
    check_seqvec_path(args.seqvec_path)
    input_sequences, ids = read_fasta(args.sequences)
    
    X_test, X_test_residue = tamper.embed_seqs(input_sequences, args.seqvec_path)
    if not os.path.isdir(args.models_dir):
        print('Please provide a valid trained model directory (--models_dir).')
        exit()

    preds = tamper.predict(X_test, X_test_residue, args.models_dir)

    if not os.path.exists(PREDICTIONS_DIR): os.makedirs(PREDICTIONS_DIR)
    outfilename = "{}{}.csv".format(PREDICTIONS_DIR, args.out)
    outf = open(outfilename, 'w+')
    outf.write("sequence,id,score\n")
    for i, score in enumerate(preds):
        outf.write("{},{},{:.5f}\n".format(input_sequences[i],ids[i], score))
    outf.close()
    print("Predictions saved in", outfilename)

def argparser():
    # Create the parser.
    my_parser = argparse.ArgumentParser(prog='tamper', 
                                        usage='%(prog)s [train|predict] [options]\n\n{}'.format(HELP_TEXT),
                                        description='Perform classification and training of protein sequences.')

    # train mode command line options.
    my_parser.add_argument("mode", type=str, help="Train or predict mode.", default="predict")
    my_parser.add_argument("--pos", type=str, help="Positive training data fasta file.")
    my_parser.add_argument("--neg", type=str, help="Negative training data fasta file.")
    my_parser.add_argument("--name", type=str, help="A name for the model to be trained.", default="new_train")
    
    # predict mode command line options.
    my_parser.add_argument("--sequences", type=str, help="Custom fasta file for prediction using a saved model.")
    my_parser.add_argument("--models_dir", type=str, 
        help="Name of the directory under which models to be ensembled for prediction are stored. Starts with models/.", default="tamper")
    my_parser.add_argument("--out", type=str, 
        help="Name of a file with which to save predictions.", default="predictions")
    
    # common command line options.
    my_parser.add_argument("--seqvec_path", type=str, 
        help="Path to trained seqvec model ( ... /uniref50_v2). Add file directory of where your SeqVec model lies.", 
        default=DEFAULT_SEQVEC_PATH)

    args = my_parser.parse_args()
    return args

def main():
    """
    There are two modes in tAMPer. 
    'train' is used for training custom data on the model.
    'predict' is used for prediction on input data.
    """
    args = argparser()
    mode = args.mode
    if mode == 'train':
        train(args)
    elif mode == 'predict':
        predict(args)
    else:
        print('Usage: python3 {0} [train|predict] [options]\n{1}'.format(sys.argv[0], HELP_TEXT))
        return

if __name__ == '__main__':
    main()