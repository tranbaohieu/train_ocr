import torch
import numpy as np
import math
import os
from PIL import Image
import cv2
from torch.nn.functional import log_softmax, softmax

from model.transformerocr import VietOCR
from model.vocab import Vocab
from model.beam import Beam

def batch_translate_beam_search(img, model, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
    # img: NxCxHxW
    model.eval()
    device = img.device
    sents = []

    with torch.no_grad():
        src = model.cnn(img)
        print(src.shap)
        memories = model.transformer.forward_encoder(src)
        for i in range(src.size(0)):
#            memory = memories[:,i,:].repeat(1, beam_size, 1) # TxNxE
            memory = model.transformer.get_memory(memories, i)
            sent = beamsearch(memory, model, device, beam_size, candidates, max_seq_length, sos_token, eos_token)
            sents.append(sent)

    sents = np.asarray(sents)

    return sents
   
def translate_beam_search(img, model, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
    # img: 1xCxHxW
    model.eval()
    device = img.device

    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src) #TxNxE
        sent = beamsearch(memory, model, device, beam_size, candidates, max_seq_length, sos_token, eos_token)

    return sent
        
def beamsearch(memory, model, device, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):    
    # memory: Tx1xE
    model.eval()

    beam = Beam(beam_size=beam_size, min_length=0, n_top=candidates, ranker=None, start_token_id=sos_token, end_token_id=eos_token)

    with torch.no_grad():
#        memory = memory.repeat(1, beam_size, 1) # TxNxE
        memory = model.transformer.expand_memory(memory, beam_size)

        for _ in range(max_seq_length):
            
            tgt_inp = beam.get_current_state().transpose(0,1).to(device)  # TxN
            decoder_outputs, memory = model.transformer.forward_decoder(tgt_inp, memory)

            log_prob = log_softmax(decoder_outputs[:,-1, :].squeeze(0), dim=-1)
            beam.advance(log_prob.cpu())
            
            if beam.done():
                break
                
        scores, ks = beam.sort_finished(minimum=1)

        hypothesises = []
        for i, (times, k) in enumerate(ks[:candidates]):
            hypothesis = beam.get_hypothesis(times, k)
            hypothesises.append(hypothesis)
    
    return [1] + [int(i) for i in hypothesises[0][:-1]]

def translate(img, model, vocab=None, max_seq_length=128, sos_token=1, eos_token=2, vis_attention=False, log_dir=None, image_name=None, seperate_signs=[':', ';']):
    "data: BxCXHxW"
    model.eval()
    device = img.device
    if vis_attention:
        log_dir = os.path.join(log_dir, image_name)
        os.makedirs(log_dir, exist_ok=True)
        img_vis = img.detach().cpu().numpy()
    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)

        translated_sentence = [[sos_token]*len(img)]
        char_probs = [[1]*len(img)]

        max_length = 0
        seperate_positions = []
        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T==eos_token, axis=1)):

            tgt_inp = torch.LongTensor(translated_sentence).to(device)
            
#            output = model(img, tgt_inp, tgt_key_padding_mask=None)
#            output = model.transformer(src, tgt_inp, tgt_key_padding_mask=None)
            output, memory, attention_weights = model.transformer.forward_decoder(tgt_inp, memory)
            output = softmax(output, dim=-1)
            output = output.to('cpu')

            values, indices  = torch.topk(output, 5)
            
            indices = indices[:, -1, 0]
            indices = indices.tolist()
            if vis_attention and vocab.decode(indices) in seperate_signs:
                seperate_position = visualize_attention_3(img_vis, attention_weights, index=max_length, log_dir=log_dir, vis=vis_attention)
                seperate_positions.append(seperate_position)
            values = values[:, -1, 0]
            values = values.tolist()
            char_probs.append(values)

            translated_sentence.append(indices)   
            max_length += 1

            del output

        translated_sentence = np.asarray(translated_sentence).T
        
        char_probs = np.asarray(char_probs).T
        char_probs = np.multiply(char_probs, translated_sentence>3)
        char_probs = np.sum(char_probs, axis=-1)/(char_probs>0).sum(-1)
    
    return translated_sentence, char_probs, seperate_positions


def visualize_attention(image, attention_weights, log_dir, index, vis=False):
    attention_weights = attention_weights.detach().cpu().numpy()
    attention_weights = attention_weights.reshape(-1, 2)
    attention_weights = attention_weights.transpose(1, 0)
    # print(attention_weights.shape)
    image = image.squeeze(0)*255
    image = image.transpose((1, 2, 0))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]
    T = attention_weights.shape[1] 
    ct = [x for x in range(T)]
    for i in range(T):
      if i == 0:
        ct[i] = 2.5
        continue
      if i == T-1:
        ct[i] = 2+(i-1-2)*4+6+((width - (2+(i-1-2)*4+6))/2)
      else:
        ct[i] = 2+(i-1)*4+3
    # stop = 23
    # for i in range(T):
    #   if i < 5:
    #     start = 4*max(i-5, 0) + 1
    #   else:
    #     start = 4*max(i-5, 0) + 2
    #   stop = min(stop + 4, width)
    #   ct[i] = (start+stop)/2
    # ct = np.array(ct)
    ct = np.expand_dims(ct, axis=0)
    ct = np.vstack((ct, ct))
    center = np.sum(ct * attention_weights)
    print(center)
    if vis:
        cv2.line(image, (int(center), 0), (int(center), height), (0, 255, 0), thickness=3, lineType=8)
        image_path = os.path.join(log_dir, '{}.jpg'.format(index))
        cv2.imwrite(image_path, image)
    return int(center)

def visualize_attention_3(image, attention_weights, log_dir, index, vis=False):
    attention_weights = attention_weights.detach().cpu().numpy()
    batch_size = attention_weights.shape[0]
    # attention_weights = attention_weights.reshape(batch_size, -1, 2)
    # attention_weights = attention_weights.transpose(batch_size, 1, 0)
    image = image.squeeze(0)*255
    image = image.transpose((1, 2, 0))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]
    T = attention_weights.shape[1] // 2
    ct = [x for x in range(T)]
    for i in range(T):
      if i == 0:
        ct[i] = 2.5
        continue
      if i == T-1:
        ct[i] = 2+(i-1-2)*4+6+((width - (2+(i-1-2)*4+6))/2)
      else:
        ct[i] = 2+(i-1)*4+3
    # stop = 23
    # for i in range(T):
    #   if i < 5:
    #     start = 4*max(i-5, 0) + 1
    #   else:
    #     start = 4*max(i-5, 0) + 2
    #   stop = min(stop + 4, width)
    #   ct[i] = (start+stop)/2
    # ct = np.array(ct)
    ct = np.expand_dims(ct, axis=0)
    ct = np.vstack((ct, ct))
    ct = ct.transpose((1, 0))
    ct = ct.flatten()
    batch_ct = np.zeros_like(attention_weights)
    for i in range(batch_size):
        batch_ct[i] = ct
    attention_weights = torch.from_numpy(attention_weights).unsqueeze(1)
    batch_ct = torch.from_numpy(batch_ct).unsqueeze(2)
    # centers = np.dot(attention_weights, batch_ct.T)
    print(attention_weights.size())
    centers = torch.bmm(attention_weights, batch_ct).numpy()
    print(centers.shape)
    if vis:
        center = centers[0]
        cv2.line(image, (int(center), 0), (int(center), height), (0, 255, 0), thickness=3, lineType=8)
        image_path = os.path.join(log_dir, '{}.jpg'.format(index))
        cv2.imwrite(image_path, image)
    return centers


def visualize_attention_2(image, attention_weights, log_dir, index):
    attention_weights = attention_weights.detach().cpu().numpy()
    attention_weights = attention_weights.squeeze(0)
    attention_weights = attention_weights.reshape(-1, 2) * 255
    attention_weights = attention_weights.transpose(1, 0)
    attention_weights = cv2.resize(attention_weights, (image.shape[3], image.shape[2]), interpolation = cv2.INTER_AREA)
    image = image.squeeze(0)*255
    image = image.transpose((1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image += attention_weights
    image = image/np.max(image) * 255
    image_path = os.path.join(log_dir, '{}.jpg'.format(index))
    cv2.imwrite(image_path, attention_weights)

def build_model(config):
    vocab = Vocab(config['vocab'])
    device = config['device']
    
    model = VietOCR(len(vocab),
            config['backbone'],
            config['cnn'], 
            config['transformer'],
            config['seq_modeling'])
    
    model = model.to(device)

    return model, vocab

def resize(w, h, expected_height, image_min_width, image_max_width):
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w/round_to)*round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w, expected_height

def process_image(image, image_height, image_min_width, image_max_width):
    img = image.convert('RGB')

    w, h = img.size
    new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)

    img = img.resize((new_w, image_height), Image.ANTIALIAS)

    img = np.asarray(img).transpose(2,0, 1)
    img = img/255
    return img

def process_input(image, image_height, image_min_width, image_max_width):
    img = process_image(image, image_height, image_min_width, image_max_width)
    img = img[np.newaxis, ...]
    img = torch.FloatTensor(img)
    return img

def predict(filename, config):
    img = Image.open(filename)
    img = process_input(img)

    img = img.to(config['device'])

    model, vocab = build_model(config)
    s = translate(img, model)[0].tolist()
    s = vocab.decode(s)
    
    return s

