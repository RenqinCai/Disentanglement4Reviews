import os
import json
import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

from ptb import PTB
from BGoogle import BGoogle
from amazon import Amazon
from utils import to_var, idx2word, expierment_name
from model import SentenceVAE

from nltk.tokenize import TweetTokenizer

def f_raw2vec(tokenizer, raw_text, w2i, i2w):
    words = tokenizer.tokenize(raw_text)
    input_text = ['<sos>']+words+['<eos>']
    input_text = [w2i.get(w, w2i['<unk>']) for w in input_text]
    print("words", words)
    return input_text

def f_test_example(model, tokenizer, w2i, i2w):
    raw_text = "since the wii wasn't being used much anymore in the living room , i thought i'd try to take it to the bedroom so i can watch hulu and stuff like that . this cable made it possible."
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
    
    mean = mean.unsqueeze(0)
    # print("mean", mean)
    # print("z", z)

    samples, z = model.inference(z=mean)
    print("<-"*10, *idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')


def main(args):

    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    splits = ['train', 'valid'] + (['test'] if args.test else [])

    datasets = OrderedDict()
    for split in splits:
        # datasets[split] = BGoogle(
        #     data_dir=args.data_dir,
        #     split=split,
        #     create_data=args.create_data,
        #     batch_size=args.batch_size ,
        #     max_sequence_length=args.max_sequence_length,
        #     min_occ=args.min_occ
        # )

        datasets[split] = Amazon(
            data_dir=args.data_dir,
            split=split,
            create_data=args.create_data,
            batch_size=args.batch_size ,
            max_sequence_length=args.max_sequence_length,
            min_occ=args.min_occ
        )

    model = SentenceVAE(
        vocab_size=datasets['train'].vocab_size,
        sos_idx=datasets['train'].sos_idx,
        eos_idx=datasets['train'].eos_idx,
        pad_idx=datasets['train'].pad_idx,
        unk_idx=datasets['train'].unk_idx,
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

    if torch.cuda.is_available():
        model = model.cuda()

    print(model)

    tokenizer = TweetTokenizer(preserve_case=False)
    vocab_file = "amazon.vocab.json"
    with open(os.path.join(args.data_dir, vocab_file), 'r') as file:
        vocab = json.load(file)
        w2i, i2w = vocab['w2i'], vocab['i2w']

    if args.tensorboard_logging:
        writer = SummaryWriter(os.path.join(args.logdir, expierment_name(args,ts)))
        writer.add_text("model", str(model))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)

    # save_model_path = os.path.join(args.save_model_path, ts)
    save_model_path = args.save_model_path

    if not os.path.exists(save_model_path):            
        os.makedirs(save_model_path)

    def kl_anneal_function(anneal_function, step, k, x0):
        if anneal_function == 'logistic':
            return float(1/(1+np.exp(-k*(step-x0))))
        elif anneal_function == 'linear':
            return min(1, step/x0)

    NLL = torch.nn.NLLLoss(size_average=False, ignore_index=datasets['train'].pad_idx)
    def loss_fn(logp, target, length, mean, logv, anneal_function, step, k, x0):

        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).data].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))
        
        # Negative Log Likelihood
        NLL_loss = NLL(logp, target)

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = kl_anneal_function(anneal_function, step, k, x0)

        return NLL_loss, KL_loss, KL_weight

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0
    save_mode = True
    last_ELBO = 1e32
    
    for epoch in range(args.epochs):
        print("+"*20)

        # f_test_example(model, tokenizer, w2i, i2w)
        for split in splits:

            # data_loader = DataLoader(
            #     dataset=datasets[split],
            #     batch_size=args.batch_size,
            #     shuffle=split=='train',
            #     num_workers=cpu_count(),
            #     pin_memory=torch.cuda.is_available()
            # )
            batch_size=args.batch_size
            tracker = defaultdict(tensor)

            # Enable/Disable Dropout
            if split == 'train':
                model.train()
            else:
                model.eval()

            # for iteration, batch in enumerate(data_loader):
            iteration = 0
            iteration_total = datasets[split].batch_num
            print("batch_num", iteration_total)
            for input_batch_tensor, target_batch_tensor, length_batch_tensor in datasets[split]:
                
                if torch.is_tensor(input_batch_tensor):
                    input_batch_tensor = to_var(input_batch_tensor)

                if torch.is_tensor(target_batch_tensor):
                    target_batch_tensor = to_var(target_batch_tensor)

                if torch.is_tensor(length_batch_tensor):
                    length_batch_tensor = to_var(length_batch_tensor)
                
                # batch_size = batch['input'].size(0)

                # for k, v in batch.items():
                #     if torch.is_tensor(v):
                #         batch[k] = to_var(v)

                # Forward pass
                # logp, mean, logv, z = model(batch['input'], batch['length'])
                logp, mean, logv, z = model(input_batch_tensor, length_batch_tensor)

                # loss calculation
                NLL_loss, KL_loss, KL_weight = loss_fn(logp, target_batch_tensor,
                    length_batch_tensor, mean, logv, args.anneal_function, step, args.k, args.x0)

                loss = (NLL_loss + KL_weight * KL_loss)/batch_size

                # backward + optimization
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1

                iteration += 1
                # bookkeepeing
                # print("elbo", tracker['ELBO'])
                # print("loss", loss)
                if iteration == 0:
                    tracker['ELBO'] = loss.data
                    tracker['ELBO'] = tracker['ELBO'].view(1)
                else:
                    tracker['ELBO'] = torch.cat((tracker['ELBO'], loss.view(1)))

                if args.tensorboard_logging:
                    # print(loss.data)
                    writer.add_scalar("%s/ELBO"%split.upper(), loss.data.item(), epoch*iteration_total + iteration)
                    writer.add_scalar("%s/NLL Loss"%split.upper(), NLL_loss.data.item()/batch_size, epoch*iteration_total + iteration)
                    writer.add_scalar("%s/KL Loss"%split.upper(), KL_loss.data.item()/batch_size, epoch*iteration_total + iteration)
                    writer.add_scalar("%s/KL Weight"%split.upper(), KL_weight, epoch*iteration_total + iteration)

                if iteration % args.print_every == 0 or iteration+1 == iteration_total:
                    print("%s Batch %04d/%i, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
                        %(split.upper(), iteration, iteration_total-1, loss.data.item(), NLL_loss.data.item()/batch_size, KL_loss.data.item()/batch_size, KL_weight))

                # if split == 'valid':
                    # if 'target_sents' not in tracker:
                    #     tracker['target_sents'] = list()
                    # tracker['target_sents'] += idx2word(batch['target'].data, i2w=datasets['train'].get_i2w(), pad_idx=datasets['train'].pad_idx)

                    # # print("z", tracker['z'], z)
                    # tracker['z'] = torch.cat((tracker['z'], z.data), dim=0)
                    # break

            print("%s Epoch %02d/%i, Mean ELBO %9.4f"%(split.upper(), epoch, args.epochs, torch.mean(tracker['ELBO'])))

            cur_ELBO = torch.mean(tracker['ELBO'])
            if args.tensorboard_logging:
                writer.add_scalar("%s-Epoch/ELBO"%split.upper(), cur_ELBO, epoch)

            if split == "valid":
                if cur_ELBO < last_ELBO:
                    save_mode = True
                else:
                    save_mode = False
                last_ELBO = cur_ELBO 

            # save a dump of all sentences and the encoded latent space
            # if split == 'valid':
            #     dump = {'target_sents':tracker['target_sents'], 'z':tracker['z'].tolist()}
            #     if not os.path.exists(os.path.join('dumps', ts)):
            #         os.makedirs('dumps/'+ts)
            #     with open(os.path.join('dumps/'+ts+'/valid_E%i.json'%epoch), 'w') as dump_file:
            #         json.dump(dump,dump_file)

            # save checkpoint
            if split == 'train':
                # checkpoint_path = os.path.join(save_model_path, "E%i.pytorch"%(epoch))
                checkpoint_path = os.path.join(save_model_path, "best.pytorch")
                if save_mode == True:
                    torch.save(model.state_dict(), checkpoint_path)
                    print("Model saved at %s"%checkpoint_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_sequence_length', type=int, default=40)
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
    parser.add_argument('-ls', '--latent_size', type=int, default=256)
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

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert args.anneal_function in ['logistic', 'linear']
    assert 0 <= args.word_dropout <= 1

    main(args)
