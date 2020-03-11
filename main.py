"""
Date: 2020-02-14
Author: figalit (github.com/figalit)
"""
import argparse
import sys
import os
import tamper

DEFAULT_SEQVEC_PATH = "seqvec/uniref50_v2" # The default path of seqvec pretrained model, change as necessary.
HELP_TEXT = '''
Use the 'train' mode to input your own positive and negative sequences, and train our model on your sequences.
Use the 'predict' mode to get predictions for your test set. Make sure to have labels in your fasta file, for downstream analysis and statistics.
'''

def read_fasta(filename):
    f = open(filename, "r+")
    lines = f.readlines()
    output = []
    for line in lines:
        if line[0] != '>': output.append(line.strip().upper())
    f.close()
    return output

def check_seqvec_path(args):
    if not os.path.isdir('./{}'.format(args.seqvec_path)):
        print("Please provide a valid seqvec trained model path (--seqvec_path). Consider adding the model under seqvec/uniref50_v2 for simplicity.")
        return 0
    else: print("Using SeqVec model path: {}".format(args.seqvec_path))

def train(args):
    if args.neg == None or args.pos == None:
        print("Please enter a positive (--pos) and negative (--neg) fasta file.")
        return 0
    
    pos_filename = args.pos
    neg_filename = args.neg
    positive_seqs = read_fasta(pos_filename)
    negative_seqs = read_fasta(neg_filename)

    check_seqvec_path(args)
    X_train, y_train = tamper.train_embed(positive_seqs, negative_seqs, args.seqvec_path)
    model = tamper.get_model()
    # print("custom_model == None: {}".format(args.custom_model ==  None))
    if args.custom_model != None:
        import shutil
        shutil.copy(args.custom_model, 'custom_model.py')
        import custom_model
        model = custom_model.get_model()
    tamper.train(model, X_train, y_train, args.name)
    # print(args)
    if args.pca_2d:
        import visualizations
        visualizations.plot_PCA(X_train,y_train)

def predict(args):
    if args.sequences == None:
        print("Please enter a fasta file of sequences for prediction.")
        return 0
    sequences_filename = args.sequences
    seqs = read_fasta(sequences_filename)
    check_seqvec_path(args)
    X_test = tamper.seqs_embed(seqs, args.seqvec_path)
    if not os.path.isfile('./models/{}.sav'.format(args.model_name)):
        print("Please provide a valid saved model path to use for prediction.")
        return 0
    tamper.predict(args.model_name, X_test, seqs)

def argparser():
    # Create the parser.
    my_parser = argparse.ArgumentParser(prog='tamper', 
                                        usage='%(prog)s [train|predict] [options]\n\n{}'.format(HELP_TEXT),
                                        description='Perform classification and training of protein sequences.')

    # train mode command line options.
    my_parser.add_argument("mode", type=str, help="Train or predict mode.", default="train")
    my_parser.add_argument("--pos", type=str, help="Positive training data fasta file.")
    my_parser.add_argument("--neg", type=str, help="Negative training data fasta file.")
    my_parser.add_argument("--name", type=str, help="A name for the model to be trained.", default="trained_model")
    my_parser.add_argument("--custom_model", type=str, help="Path to a python file with a classification model definition under a get_model function.")
    my_parser.add_argument("--seqvec_path", type=str, help="Path to trained seqvec model (.../uniref50_v2).", default=DEFAULT_SEQVEC_PATH)
    my_parser.add_argument("--pca_2d", type=bool, help="Whether or not to display PCA dimensionality reduction to 2 dimensions.", default=False)

    # predict mode command line options.
    my_parser.add_argument("--sequences", type=str, help="Custom fasta file for prediction using a saved model.")
    my_parser.add_argument("--model_name", type=str, help="Name of the model to use for predictions. Model should be under models/ dir.", default="tamper_saved_model")
    args = my_parser.parse_args()
    return args

def main():
    """
    There are two main modes in Toxoria. 
    'train' is used for training custom data on the model.
    'predict' is used for prediction of data.
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
    # TODO(figalit): Also store the embeddings of protein sequences.