from optim.optim import ScheduledOptim
from optim.labelsmoothingloss import LabelSmoothingLoss
from torch.optim import Adam, SGD, AdamW
from torch import nn
from tool.translate import build_model
from tool.translate import translate, batch_translate_beam_search
from tool.utils import download_weights
from tool.logger import Logger
from loader.aug import ImgAugTransform
from model.vocab import Vocab
from model.seqmodel.seq2seq import Seq2Seq, Decoder, Attention

import yaml
import torch
from loader.dataloader_v1 import DataGen
from loader.dataloader_v2 import DataGenV2
from loader.dataloader import OCRDataset, ClusterRandomSampler, Collator
from torch.utils.data import DataLoader
from einops import rearrange
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, OneCycleLR

import torchvision 

from tool.utils import compute_accuracy
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import time

class Trainer():
    def __init__(self, config, pretrained=False, augmentor=ImgAugTransform()):

        self.config = config
        self.model, self.vocab = build_model(config)
        # self.vocab_new = Vocab(self.config['vocab'] + ' ' + '£€¥')
        
        self.device = config['device']
        self.num_iters = config['trainer']['iters']
        self.beamsearch = config['predictor']['beamsearch']

        self.data_root = config['dataset']['data_root']
        self.train_annotation = config['dataset']['train_annotation']
        self.valid_annotation = config['dataset']['valid_annotation']
        self.dataset_name = config['dataset']['name']
        self.data_root_test = config['dataset']['data_root_test'] 
        self.batch_size = config['trainer']['batch_size']

        self.batch_size = config['trainer']['batch_size']
        self.print_every = config['trainer']['print_every']
        self.valid_every = config['trainer']['valid_every']
        
        self.image_aug = config['aug']['image_aug']
        self.masked_language_model = config['aug']['masked_language_model']

        self.checkpoint = config['trainer']['checkpoint']
        self.export_weights = config['trainer']['export']
        self.metrics = config['trainer']['metrics']
        logger = config['trainer']['log']
    
        if logger:
            self.logger = Logger(logger) 

        if pretrained:
            weight_file = download_weights(**config['pretrain'], quiet=config['quiet'])
            self.load_weights(weight_file)

        self.iter = 0
        
        self.optimizer = AdamW(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09)
        self.scheduler = OneCycleLR(self.optimizer, total_steps=self.num_iters, **config['optimizer'])
#        self.optimizer = ScheduledOptim(
#            Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09),
#            #config['transformer']['d_model'], 
#            512,
#            **config['optimizer'])

        self.criterion = LabelSmoothingLoss(len(self.vocab), padding_idx=self.vocab.pad, smoothing=0.1)
        
        transforms = None
        if self.image_aug:
            transforms =  augmentor

        # self.train_gen = self.data_gen('train_{}'.format(self.dataset_name), 
        #         self.data_root, self.train_annotation, self.masked_language_model, transform=transforms)
        self.train_gen = self.data_gen_v2(data_root=self.data_root, train=True)
        if self.valid_annotation:
            self.valid_gen = self.data_gen_v2(data_root=self.data_root_test, train=False)
            self.test_gen = self.data_gen_v2(data_root=["./OCR/public_test"], train=False)

        self.train_losses = []
        
    def train(self):
        total_loss = 0
        
        total_loader_time = 0
        total_gpu_time = 0
        best_acc = 0
        best_score = 0

        data_iter = self.train_gen.gen(batch_size=self.batch_size)
        for i in range(self.num_iters):
            self.iter += 1

            start = time.time()

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = self.train_gen.gen(batch_size=self.batch_size)
                batch = next(data_iter)

            total_loader_time += time.time() - start

            start = time.time()
            loss = self.step(batch)
            total_gpu_time += time.time() - start

            total_loss += loss
            self.train_losses.append((self.iter, loss))

            if self.iter % self.print_every == 0:
                info = 'iter: {:06d} - train loss: {:.3f} - lr: {:.2e} - load time: {:.2f} - gpu time: {:.2f}'.format(self.iter, 
                        total_loss/self.print_every, self.optimizer.param_groups[0]['lr'], 
                        total_loader_time, total_gpu_time)

                total_loss = 0
                total_loader_time = 0
                total_gpu_time = 0
                print(info) 
                self.logger.log(info)

            if self.valid_annotation and self.iter % self.valid_every == 0:
                # val_loss = self.validate()
                acc_full_seq, acc_per_char, acc_lev_score = self.precision(self.metrics, valid=True)
                acc_full_seq_test, acc_per_char_test, acc_lev_score_test = self.precision(self.metrics, valid=False)

                info = 'iter: {:06d} - acc full seq: {:.4f} - acc per char: {:.4f} - acc lev score: {:.4f} - acc full seq test: {:.4f} - acc per char test: {:.4f} - acc lev score test: {:.4f}'.format(self.iter, acc_full_seq, acc_per_char, acc_lev_score, acc_full_seq_test, acc_per_char_test, acc_lev_score_test)
                print(info)
                self.logger.log(info)

                if acc_lev_score_test > best_score:
                    name_checkpoint = 'vgg_seq2seq_score_{:.4f}.pth'.format(acc_lev_score_test)
                    self.save_weights(os.path.join(self.export_weights, name_checkpoint))
                    best_score = acc_lev_score_test

                if acc_full_seq > best_acc:
                    name_checkpoint = 'vgg_seq2seq_acc_{:.4f}.pth'.format(acc_full_seq)
                    self.save_weights(os.path.join(self.export_weights, name_checkpoint))
                    best_acc = acc_full_seq
                self.save_weights(os.path.join(self.export_weights, "last.pth"))

            
    def validate(self):
        self.model.eval()

        total_loss = []
        
        with torch.no_grad():
            self.valid_gen_loader = self.valid_gen.gen(batch_size=self.batch_size, shuffle=False)
            for step, batch in enumerate(self.valid_gen_loader):
                batch = self.batch_to_device(batch)
                img, tgt_input, tgt_output, tgt_padding_mask = batch['img'], batch['tgt_input'], batch['tgt_output'], batch['tgt_padding_mask']

                outputs = self.model(img, tgt_input, tgt_padding_mask)
#                loss = self.criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_output, 'b o -> (b o)'))
               
                outputs = outputs.flatten(0,1)
                tgt_output = tgt_output.flatten()
                loss = self.criterion(outputs, tgt_output)

                total_loss.append(loss.item())
                
                del outputs
                del loss

        total_loss = np.mean(total_loss)
        self.model.train()
        
        return total_loss
    
    def predict(self, sample=None, valid=True):
        pred_sents = []
        actual_sents = []
        img_files = []

        if valid:
            self.valid_gen_loader = self.valid_gen.gen(batch_size=self.batch_size)
        else:
            self.valid_gen_loader = self.test_gen.gen(batch_size=self.batch_size)

        for batch in  self.valid_gen_loader:
            batch = self.batch_to_device(batch)

            if self.beamsearch:
                translated_sentence = batch_translate_beam_search(batch['img'], self.model)
                prob = None
            else:
                translated_sentence, prob, _ = translate(batch['img'], self.model)

            pred_sent = self.vocab.batch_decode(translated_sentence.tolist())
            actual_sent = self.vocab.batch_decode(batch['tgt_output'].tolist())

            img_files.extend(batch['filenames'])

            pred_sents.extend(pred_sent)
            actual_sents.extend(actual_sent)
            
            if sample != None and len(pred_sents) > sample:
                break

        return pred_sents, actual_sents, img_files, prob

    def precision(self, sample=None, valid=True):

        pred_sents, actual_sents, _, _ = self.predict(sample=sample, valid=valid)

        acc_full_seq = compute_accuracy(actual_sents, pred_sents, mode='full_sequence')
        acc_per_char = compute_accuracy(actual_sents, pred_sents, mode='per_char')
        acc_lev_score = compute_accuracy(actual_sents, pred_sents, mode='lev_score')
    
        return acc_full_seq, acc_per_char, acc_lev_score
    
    def visualize_prediction(self, sample=16, errorcase=False, fontname='serif', fontsize=16):
        
        pred_sents, actual_sents, img_files, probs = self.predict(sample)

        if errorcase:
            wrongs = []
            for i in range(len(img_files)):
                if pred_sents[i]!= actual_sents[i]:
                    wrongs.append(i)

            pred_sents = [pred_sents[i] for i in wrongs]
            actual_sents = [actual_sents[i] for i in wrongs]
            img_files = [img_files[i] for i in wrongs]
            probs = [probs[i] for i in wrongs]

        img_files = img_files[:sample]

        fontdict = {
                'family':fontname,
                'size':fontsize
                } 

        for vis_idx in range(0, len(img_files)):
            img_path = img_files[vis_idx]
            pred_sent = pred_sents[vis_idx]
            actual_sent = actual_sents[vis_idx]
            prob = probs[vis_idx]

            img = Image.open(open(img_path, 'rb'))
            plt.figure()
            plt.imshow(img)
            plt.title('prob: {:.3f} - pred: {} - actual: {}'.format(prob, pred_sent, actual_sent), loc='left', fontdict=fontdict)
            plt.axis('off')

        plt.show()
    
    def visualize_dataset(self, sample=16, fontname='serif'):
        n = 0
        for batch in self.train_gen:
            for i in range(self.batch_size):
                img = batch['img'][i].numpy().transpose(1,2,0)
                sent = self.vocab.decode(batch['tgt_input'].T[i].tolist())
                
                plt.figure()
                plt.title('sent: {}'.format(sent), loc='center', fontname=fontname)
                plt.imshow(img)
                plt.axis('off')
                
                n += 1
                if n >= sample:
                    plt.show()
                    return


    def load_checkpoint(self, filename, use_freeze=False):
        checkpoint = torch.load(filename)
        
        if use_freeze:
            cnn_state_dict = {}
            for item in checkpoint.items():
                key, value = item
                if key.split('.')[0] == 'cnn':
                    new_key = key.replace('cnn.', '')
                    cnn_state_dict.update({new_key : value})
            self.model.cnn.load_state_dict(cnn_state_dict)
            for param in self.model.cnn.parameters():
                param.requires_grad = False
            encoder_state_dict = {}
            for item in checkpoint.items():
                key, value = item
                if 'transformer.encoder' in key:
                    new_key = key.replace('transformer.encoder.', '')
                    encoder_state_dict.update({new_key : value})
            self.model.transformer.encoder.load_state_dict(encoder_state_dict)
            for param in self.model.transformer.encoder.parameters():
                param.requires_grad = False
        else:
            self.model.load_state_dict(checkpoint)

    def save_checkpoint(self, filename):
        state = {'iter':self.iter, 'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'train_losses': self.train_losses}
        
        path, _ = os.path.split(filename)
        os.makedirs(path, exist_ok=True)

        torch.save(state, filename)

    def load_weights(self, filename):
        state_dict = torch.load(filename, map_location=torch.device(self.device))

        for name, param in self.model.named_parameters():
            if name not in state_dict:
                print('{} not found'.format(name))
            elif state_dict[name].shape != param.shape:
                print('{} missmatching shape, required {} but found {}'.format(name, param.shape, state_dict[name].shape))
                del state_dict[name]

        self.model.load_state_dict(state_dict, strict=False)

    def save_weights(self, filename):
        path, _ = os.path.split(filename)
        os.makedirs(path, exist_ok=True)
       
        torch.save(self.model.state_dict(), filename)

    def batch_to_device(self, batch):
        img = batch['img'].to(self.device, non_blocking=True)
        tgt_input = batch['tgt_input'].to(self.device, non_blocking=True)
        tgt_output = batch['tgt_output'].to(self.device, non_blocking=True)
        tgt_padding_mask = batch['tgt_padding_mask'].to(self.device, non_blocking=True)

        batch = {
                'img': img, 'tgt_input':tgt_input, 
                'tgt_output':tgt_output, 'tgt_padding_mask':tgt_padding_mask, 
                'filenames': batch['filenames']
                }

        return batch

    def data_gen(self, lmdb_path, data_root, annotation, masked_language_model=True, transform=None):
        dataset = OCRDataset(lmdb_path=lmdb_path, 
                root_dir=data_root, annotation_path=annotation, 
                vocab=self.vocab, transform=transform, 
                image_height=self.config['dataset']['image_height'], 
                image_min_width=self.config['dataset']['image_min_width'], 
                image_max_width=self.config['dataset']['image_max_width'])

        sampler = ClusterRandomSampler(dataset, self.batch_size, True)
        collate_fn = Collator(masked_language_model)

        gen = DataLoader(
                dataset,
                batch_size=self.batch_size, 
                sampler=sampler,
                collate_fn = collate_fn,
                shuffle=False,
                drop_last=False,
                **self.config['dataloader'])
       
        return gen

    def data_gen_v1(self, lmdb_path, data_root, annotation):
        data_gen = DataGen(data_root, annotation, self.vocab, 'cpu', 
                image_height = self.config['dataset']['image_height'],        
                image_min_width = self.config['dataset']['image_min_width'],
                image_max_width = self.config['dataset']['image_max_width'])

        return data_gen

    def data_gen_v2(self, data_root, train):
        data_gen = DataGenV2(data_root, self.vocab, self.device, train=train, 
                image_height = self.config['dataset']['image_height'],        
                image_min_width = self.config['dataset']['image_min_width'],
                image_max_width = self.config['dataset']['image_max_width'])

        return data_gen

    def step(self, batch):
        self.model.train()

        batch = self.batch_to_device(batch)
        img, tgt_input, tgt_output, tgt_padding_mask = batch['img'], batch['tgt_input'], batch['tgt_output'], batch['tgt_padding_mask']    
        
        outputs = self.model(img, tgt_input, tgt_key_padding_mask=tgt_padding_mask)
#        loss = self.criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_output, 'b o -> (b o)'))
        outputs = outputs.view(-1, outputs.size(2))#flatten(0, 1)
        tgt_output = tgt_output.view(-1)#flatten()
        
        loss = self.criterion(outputs, tgt_output)

        self.optimizer.zero_grad()

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1) 

        self.optimizer.step()
        self.scheduler.step()

        loss_item = loss.item()

        return loss_item
