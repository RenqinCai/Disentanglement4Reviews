import os
import json
import time
import torch
import argparse
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from model import *
from dataset import *
from train import *
from evaluation import *
from metric import *

def main(args):

    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    splits = ['train', 'valid']

    train_seq_corpus, valid_seq_corpus, test_seq_corpus = RawData(data_dir=args.data_dir, split_ratio=split_ratio, create_data=args.create_data, max_sequence_length=args.max_sequence_length, min_occ=args.min_occ)
    
    datasets['train'] = train_seq_corpus
    datasets['valid'] = valid_seq_corpus

    model = SeqVAE(vocab_size=datasets['train'].vocab_size, sos_idx=datasets['train'].sos_idx, eos_idx=datasets['train'].eos_idx, pad_idx=datasets['train'].pad_idx, unk_idx=datasets['train'].unk_idx, max_sequence_length=args.max_sequence_length, embedding_size=args.embedding_size, rnn_type=args.rnn_type, hidden_size=args.hidden_size, word_dropout=args.word_dropout, embedding_dropout=args.embedding_dropout, latent_size=args.latent_size, num_layers=args.num_layers, bidirectional=args.bidirectional)

    if torch.cuda.is_available():
        model = model.cuda()

    print(model)

    if args.tensorboard_logging:
        writer = SummaryWriter(os.path.join(args.logdir, experiment_name(args, ts)))
        writer.add_text('model', str(model))
        writer.add_text('args', str(args))
        writer.add_text("ts", ts)

    save_model_path = args.save_model_path

    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    loss_func = torch.nn.NLLLoss(size_average=False, ignore_index=datasets['train'].pad_idx)

    for epoch in range(args.epochs):
        ### train 
        train(model, optimizer, loss_func, args)

        ### test
        eval(model, optimizer, loss_func, args)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--create_data', default=True)
    parser.add_argument('--max_sequence_length', type=int, default=60)
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('--test', action='store_true')

    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)

    parser.add_argument('-af', '--anneal_function', type=str, default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)

    parser.add_argument('-v','--print_every', type=int, default=50)
    parser.add_argument('-tb','--tensorboard_logging', action='store_true')
    parser.add_argument('-log','--logdir', type=str, default='logs')
    parser.add_argument('-bin','--save_model_path', type=str, default='bin')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()
    args.anneal_function = args.anneal_function.lower()

    main(args)