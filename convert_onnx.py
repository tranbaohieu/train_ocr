import matplotlib.pyplot as plt
from PIL import Image
from tool.config import Cfg
from tool.translate import build_model, process_input, translate
import torch
import onnxruntime
import numpy as np

config = Cfg.load_config_from_file('/data/workdir/hieutb1/text_recognition_document/vietocr/config.yml')
config['cnn']['pretrained']=False
config['device'] = 'cpu'
config['vocab'] += ' '
model, vocab = build_model(config)
weight_path = '/data/workdir/hieutb1/text_recognition_document/vietocr/weights_new_chars/vgg_seq2seq_0.9077.pth'
# load weight
model.load_state_dict(torch.load(weight_path, map_location=torch.device(config['device'])))
model = model.eval() 

def convert_cnn_part(img, save_path, model, max_seq_length=128, sos_token=1, eos_token=2): 
    with torch.no_grad(): 
        src = model.cnn(img)
        torch.onnx.export(model.cnn, img, save_path, export_params=True, 
                        opset_version=12, do_constant_folding=True, verbose=True, 
                        input_names=['img'], output_names=['output'], 
                        dynamic_axes={'img': {0: 'batch', 1: 'channel', 2:'height', 3: 'width'}, 
                                        'output': {0: 'channel', 1: 'batch'}})
    
    return src

def convert_encoder_part(model, src, save_path): 
    encoder_outputs, hidden = model.transformer.encoder(src) 
    torch.onnx.export(model.transformer.encoder, src, save_path, export_params=True, 
                    opset_version=11, do_constant_folding=True, input_names=['src'], 
                    output_names=['encoder_outputs', 'hidden'], 
                    dynamic_axes={'src':{0: "channel_input", 1:"batch"}, 
                                    'encoder_outputs': {0: 'channel_output', 1:'batch'},
                                    'hidden': {0: 'batch'}}) 
    return hidden, encoder_outputs

def convert_decoder_part(model, tgt, hidden, encoder_outputs, save_path):
    tgt = tgt[-1]
    
    torch.onnx.export(model.transformer.decoder,
        (tgt, hidden, encoder_outputs),
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['tgt', 'hidden', 'encoder_outputs'],
        output_names=['output', 'hidden_out', 'last'],
        dynamic_axes={'tgt': {0:'batch'},
                    'encoder_outputs':{0: "channel_input", 1:'batch'},
                    'hidden': {0: 'batch'},
                    'output': {0:'batch'},
                    'hidden_out': {0 : 'batch'},
                    'last': {0: 'batch'}})

def translate_onnx(img, session, max_seq_length=128, sos_token=1, eos_token=2):
    """data: BxCxHxW"""
    cnn_session, encoder_session, decoder_session = session
    
    # create cnn input
    cnn_input = {cnn_session.get_inputs()[0].name: img}
    src = cnn_session.run(None, cnn_input)
    
    # create encoder input
    encoder_input = {encoder_session.get_inputs()[0].name: src[0]}
    encoder_outputs, hidden = encoder_session.run(None, encoder_input)
    translated_sentence = [[sos_token] * len(img)]
    max_length = 0

    while max_length <= max_seq_length and not all(
        np.any(np.asarray(translated_sentence).T == eos_token, axis=1)
    ):
        tgt_inp = translated_sentence
        decoder_input = {decoder_session.get_inputs()[0].name: tgt_inp[-1], decoder_session.get_inputs()[1].name: hidden, decoder_session.get_inputs()[2].name: encoder_outputs}

        output, hidden, attention_weights = decoder_session.run(None, decoder_input)
        output = np.expand_dims(output, axis=1)
        output = torch.Tensor(output)

        values, indices = torch.topk(output, 1)
        indices = indices[:, -1, 0]
        indices = indices.tolist()

        translated_sentence.append(indices)
        max_length += 1

        del output

    translated_sentence = np.asarray(translated_sentence).T

    return translated_sentence

def main():
    # img = torch.rand(1, 3, 32, 475)
    # src = convert_cnn_part(img, '/data/workdir/hieutb1/idp-onprem-v2/app/checkpoints/cnn.onnx', model)
    # hidden, encoder_outputs = convert_encoder_part(model, src, '/data/workdir/hieutb1/idp-onprem-v2/app/checkpoints/encoder.onnx')
    # tgt = torch.LongTensor([[1] * len(img)])
    # convert_decoder_part(model, tgt, hidden, encoder_outputs, '/data/workdir/hieutb1/idp-onprem-v2/app/checkpoints/decoder.onnx')

    img = Image.open('/data/workdir/hieutb1/text_recognition_document/vietocr/test_ocr.png')
    img = process_input(img, config['dataset']['image_height'], 
                    config['dataset']['image_min_width'], config['dataset']['image_max_width'])  
    img = img.to(config['device'])

    # create inference session
    cnn_session = onnxruntime.InferenceSession("/data/workdir/hieutb1/idp-onprem-v2/app/checkpoints/cnn.onnx", providers=['CUDAExecutionProvider'])
    encoder_session = onnxruntime.InferenceSession("/data/workdir/hieutb1/idp-onprem-v2/app/checkpoints/encoder.onnx", providers=['CUDAExecutionProvider'])
    decoder_session = onnxruntime.InferenceSession("/data/workdir/hieutb1/idp-onprem-v2/app/checkpoints/decoder.onnx", providers=['CUDAExecutionProvider'])


    session = (cnn_session, encoder_session, decoder_session)
    s = translate_onnx(np.array(img.cpu()), session)[0].tolist()
    s = vocab.decode(s)
    print(s)

if __name__ == '__main__':
    main()
    # img = torch.rand(1, 3, 32, 100)
    # print(model.cnn(img).size())