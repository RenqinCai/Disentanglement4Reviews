import os
import io
import json
import torch

import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset

# from utils import OrderedCounter
import random

random.seed(3)

class RawData():
    def __init__(self, data_dir, split_ratio, create_data, max_seq_length, min_occ):
        super().__init__()

        self.data_dir = data_dir
        self.split_ratio = split_ratio
        self.max_seq_length = max_seq_length
        self.min_occ = min_occ

        self.data_file = "_"+"item_time.pickle"
        self.raw_data_path = os.path.join(data_dir, self.data_file)  
        
        # self.vocab_file = data_name+"_"+"vocab.json"

        self.w2i = dict()
        self.i2w = dict()
        self.w2c = dict()

        if create_data:
            return self._create_data()
        else:
            return self._load_data()

    def _create_data(self):

        ### load pickle data
        action_f = open(self.raw_data_path, "rb")
        action_total = pickle.load(action_f)
        action_seq_num = len(action_total)

        print("action seq num", action_seq_num)

        train_seq_num = int(action_seq_num*self.split_ratio)

        valid_seq_num = int(action_seq_num*(1-self.split_ratio)/2)
        test_seq_num = action_seq_num-train_seq_num-valid_seq_num

        random.shuffle(action_total)

        train_seq_list = action_total[: train_seq_num]
        valid_seq_list = action_total[train_seq_num: train_seq_num+valid_seq_num]
        test_seq_list = action_total[train_seq_num+valid_seq_num: ]

        self.create_vocab(train_seq_list)

        train_seq_corpus = self.f_create_seq_corpus(train_seq_list)
        valid_seq_corpus = self.f_create_seq_corpus(valid_seq_list)
        test_seq_corpus = self.f_create_seq_corpus(test_seq_list)

        return train_seq_corpus, valid_seq_corpus, test_seq_corpus

    ### create vocab from train seq list
    def f_create_vocab(self, seq_list):
        for seq_index, seq in enumerate(seq_list):
            for action_index, action in enumerate(seq):
                if action not in self.w2c:
                    
                    self.w2c[action] = 0

                self.w2c[action] += 1

        self.w2i['sos'] = 0
        self.w2i['eos'] = 1
        self.w2i['pad'] = 2
        self.w2i['unk'] = 3

        self.i2w[0] = 'sos'
        self.i2w[1] = 'eos'
        self.i2w[2] = 'pad'
        self.i2w[3] = 'unk'

        for w in self.w2c:
            w_num = self.w2c[w]
            if w_num < self.min_occ:
                continue

            w_i = len(self.w2i)
            self.w2i[w] = w_i
            self.i2w[w_i] = w

    def f_create_seq_corpus(self, seq_list):
        seq_corpus_obj = SeqCorpus()
        seq_corpus_obj.f_set_vocab(len(self.w2i), self.w2i['sos'], self.w2i['eos'], self.w2i['pad'], self.w2i['unk'])

        for seq_index, seq in enumerate(seq_list):
            input_seq = ['sos'] + seq
            input_seq = input_seq[:self.max_seq_length]

            output_seq = seq[:self.max_seq_length-1]
            output_seq = output_seq + ['eos']

            seq_len = len(input_seq)
            input_seq.extend(['<pad>'])*(self.max_seq_length-seq_len)
            output_seq.extend(['<pad>'])*(self.max_seq_length-seq_len)

            new_input_seq = []
            new_output_seq = []

            for action_index, action in enumerate(input_seq):
                # if action not in self.w2i:
                #     continue
                new_input_seq.append(self.w2i.get(action, self.w2i['unk']))

            for action_index, action in enumerate(output_seq):
                # if action not in self.w2i:
                #     continue
                new_output_seq.append(self.w2i.get(action, self.w2i['unk']))
 
            seq_corpus_obj.f_add_seq_data(seq_index, new_input_seq, new_output_seq, seq_len)

        return seq_corpus_obj

class SeqCorpus(Dataset):
    def __init__(self):
        self.data = defaultdict(dict)
        self.vocab_size = -1
        self.sos_idx = -1
        self.eos_idx = -1
        self.pad_idx = -1
        self.unk_idx = -1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)

        return {'input': np.asarray(self.data[idx]['input']), 'target': np.asarray(self.data[idx]['target']), 'length': self.data[idx]['length']}

    def f_add_seq_data(self, input_seq, output_seq, seq_len):
        seq_index = len(data)
        data[seq_index]['input'] = input_seq
        data[seq_index]['output'] = output_seq
        data[seq_index]['length'] = seq_len

    def f_set_vocab(self, vocab_size, sos_idx, eos_idx, pad_idx, unk_idx):
        self.vocab_size = vocab_size
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
