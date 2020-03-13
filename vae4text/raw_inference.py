import os
import json
import torch
import argparse

from model import SentenceVAE
from utils import to_var, idx2word, interpolate
from torch.utils.data import DataLoader
from ptb import PTB
from amazon import Amazon
from multiprocessing import cpu_count
from nltk.tokenize import TweetTokenizer

def f_raw2vec(tokenizer, raw_text, w2i, i2w):
    words = tokenizer.tokenize(raw_text)
    input_text = ['<sos>']+words+['<eos>']
    input_text = [w2i.get(w, w2i['<unk>']) for w in input_text]
    print("words", words)
    return input_text

def main(args):

    data_name = args.data_name
    with open(args.data_dir+data_name+'.vocab.json', 'r') as file:
        vocab = json.load(file)

    w2i, i2w = vocab['w2i'], vocab['i2w']

    model = SentenceVAE(
        vocab_size=len(w2i),
        sos_idx=w2i['<sos>'],
        eos_idx=w2i['<eos>'],
        pad_idx=w2i['<pad>'],
        unk_idx=w2i['<unk>'],
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
        )

    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s"%(args.load_checkpoint))

    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()

    # samples, z = model.inference(n=args.num_samples)
    # print('----------SAMPLES----------')
    # print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')
    
    # z1 = torch.randn([args.latent_size]).numpy()
    # z2 = torch.randn([args.latent_size]).numpy()
    # z = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float())
    # samples, _ = model.inference(z=z)
    # print('-------INTERPOLATION-------')
    # print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

    # print('-------Encode ... Decode-------')
    
    # datasets = Amazon(
    #         data_dir=args.data_dir,
    #         split="valid",
    #         create_data=False,
    #         batch_size=10,
    #         max_sequence_length=args.max_sequence_length,
    #         min_occ=3
    #     )


    ### load vocab
    # with open(os.path.join(args.data_dir, args.vocab_file), 'r') as file:
    #     vocab = json.load(file)
    #     w2i, i2w = vocab['w2i'], vocab['i2w']

    tokenizer = TweetTokenizer(preserve_case=False)

    # raw_text = "I like this!"
    raw_text = "DON'T CARE FOR IT.  GAVE IT AS A GIFT AND THEY WERE OKAY WITH IT.  JUST NOT WHAT I EXPECTED."
    input_text = f_raw2vec(tokenizer, raw_text, w2i, i2w)
    length_text = len(input_text)
    length_text = [length_text]
    print("length_text", length_text)

    input_tensor = torch.LongTensor(input_text)
    print('input_tensor', input_tensor)
    input_tensor = input_tensor.unsqueeze(0)
    if torch.is_tensor(input_tensor):
        input_tensor = to_var(input_tensor)

    length_tensor = torch.LongTensor(length_text)
    print("length_tensor", length_tensor)
    # length_tensor = length_tensor.unsqueeze(0)
    if torch.is_tensor(length_tensor):
        length_tensor = to_var(length_tensor)
    
    print("*"*10)
    print("->"*10, *idx2word(input_tensor, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')
    logp, mean, logv, z = model(input_tensor, length_tensor)

    # print("z", z.size(), mean_z.size())
    mean = mean.unsqueeze(0)
    print("mean", mean)
    print("z", z)

    samples, z = model.inference(z=mean)
    print("<-"*10, *idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

    for i in range(10):
        samples, z = model.inference(z=z)
        print("<-"*10, *idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')


    # iteration = 0
    # for input_batch_tensor, target_batch_tensor, length_batch_tensor in datasets:
    #     if torch.is_tensor(input_batch_tensor):
    #         input_batch_tensor = to_var(input_batch_tensor)

    #     if torch.is_tensor(target_batch_tensor):
    #         target_batch_tensor = to_var(target_batch_tensor)

    #     if torch.is_tensor(length_batch_tensor):
    #         length_batch_tensor = to_var(length_batch_tensor)

    #     print("*"*10)
    #     print("->"*10, *idx2word(input_batch_tensor, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')
    #     logp, mean, logv, z = model(input_batch_tensor,length_batch_tensor)

    #     samples, z = model.inference(z=z)
    #     print("<-"*10, *idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')
    #     # print("+"*10)
    #     if iteration == 0:
    #         break

    #     iteration += 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str)
    parser.add_argument('-n', '--num_samples', type=int, default=10)

    parser.add_argument('-dd', '--data_dir', type=str, default='data')
    parser.add_argument('-dn', '--data_name', type=str, default='ptb')
    parser.add_argument('-v', '--vocab_file', type=str, default='amazon.vocab.json')
    parser.add_argument('-ms', '--max_sequence_length', type=int, default=50)
    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)
    parser.add_argument('-ls', '--latent_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert 0 <= args.word_dropout <= 1

    main(args)
