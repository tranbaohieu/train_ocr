import torch
import numpy as np
from PIL import Image
import random
from model.vocab import Vocab
from tool.translate import process_image
import os
import glob
from collections import defaultdict
import math
from prefetch_generator import background

class BucketData(object):
    def __init__(self, device):
        self.max_label_len = 0
        self.data_list = []
        self.label_list = []
        self.file_list = []
        self.device = device

    def append(self, datum, label, filename):
        self.data_list.append(datum)
        self.label_list.append(label)
        self.file_list.append(filename)
        
        self.max_label_len = max(len(label), self.max_label_len)

        return len(self.data_list)

    def flush_out(self):                           
        """
        Shape:
            - img: (N, C, H, W) 
            - tgt_input: (T, N) 
            - tgt_output: (N, T) 
            - tgt_padding_mask: (N, T) 
        """
        # encoder part
        img = np.array(self.data_list, dtype=np.float32)
        
        # decoder part
        target_weights = []
        tgt_input = []
        for label in self.label_list:
            label_len = len(label)
            
            tgt = np.concatenate((
                label,
                np.zeros(self.max_label_len - label_len, dtype=np.int32)))
            tgt_input.append(tgt)
            
            one_mask_len = label_len - 1
            
            target_weights.append(np.concatenate((
                np.ones(one_mask_len, dtype=np.float32),
                np.zeros(self.max_label_len - one_mask_len,dtype=np.float32))))

        # reshape to fit input shape
        tgt_input = np.array(tgt_input, dtype=np.int64).T
        tgt_output = np.roll(tgt_input, -1, 0).T
        tgt_output[:, -1]=0
        
        tgt_padding_mask = np.array(target_weights)==0

        filenames = self.file_list

        self.data_list, self.label_list, self.file_list = [], [], []
        self.max_label_len = 0
        
        rs = {
            'img': torch.FloatTensor(img).to(self.device),
            'tgt_input': torch.LongTensor(tgt_input).to(self.device),
            'tgt_output': torch.LongTensor(tgt_output).to(self.device),
            'tgt_padding_mask':torch.BoolTensor(tgt_padding_mask).to(self.device),
            'filenames': filenames
        }
        
        return rs

    def __len__(self):
        return len(self.data_list)

    def __iadd__(self, other):
        self.data_list += other.data_list
        self.label_list += other.label_list
        self.max_label_len = max(self.max_label_len, other.max_label_len)
        self.max_width = max(self.max_width, other.max_width)

    def __add__(self, other):
        res = BucketData()
        res.data_list = self.data_list + other.data_list
        res.label_list = self.label_list + other.label_list
        res.max_width = max(self.max_width, other.max_width)
        res.max_label_len = max((self.max_label_len, other.max_label_len))
        return res

class DataGenV2(object):

    def __init__(self, data_roots, vocab, device, train=True, image_height=32, image_min_width=32, image_max_width=512):
        self.image_height = image_height
        self.image_min_width = image_min_width
        self.image_max_width = image_max_width
        self.image_path_list = []
        self.label_list = []
        self.train = True
        
        for data_root in data_roots:
            if train:
                label_filename = "train_labels.txt"
            else:
                label_filename = "val_labels.txt"
            label_path = os.path.join(data_root, label_filename)
            syn_data = False
            if not os.path.exists(label_path):
                label_path = os.path.join(data_root, "labels.txt")
                syn_data = True
            lines = open(label_path, 'r').readlines()[:-1]
            for line in lines:
                # try:
                image_name, label = line.strip().split(maxsplit=1)
                label = label.strip()
                label = label.replace("”", "\"")
                if syn_data:
                    image_name = os.path.join(data_root, image_name)
                self.image_path_list.append(image_name)
                self.label_list.append(label)
        self.nSamples = len(self.image_path_list)
        print(self.nSamples)
        self.vocab = vocab
        self.device = device
        self.train = train
        
        self.clear()

    def clear(self):
        self.bucket_data = defaultdict(lambda: BucketData(self.device))

    @background(max_prefetch=1) 
    def gen(self, batch_size, last_batch=True, shuffle=True):
        idx_list = [i for i in range(self.nSamples)]
        if shuffle:
            np.random.shuffle(idx_list)
        for i in idx_list:     
            
            img_path, lex = self.image_path_list[i], self.label_list[i]
            
            try:
                img_bw, word = self.read_data(img_path, lex)
            except IOError:
                print('ioread image:{}'.format(img_path))
                
            width = img_bw.shape[-1]

            bs = self.bucket_data[width].append(img_bw, word, img_path)
            if bs >= batch_size:
                b = self.bucket_data[width].flush_out()
                yield b

        if last_batch: 
            for bucket in self.bucket_data.values():
                if len(bucket) > 0:
                    b = bucket.flush_out()
                    yield b

        self.clear()

    def read_data(self, img_path, lex):        
        
        with open(img_path, 'rb') as img_file:
            img = Image.open(img_file).convert('RGB')
            img_bw = process_image(img, self.image_height, self.image_min_width, self.image_max_width)
        
        word = self.vocab.encode(lex)

        return img_bw, word

