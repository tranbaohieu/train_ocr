import argparse

from model.trainer import Trainer
from tool.config import Cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config.yml', help='see example at ')
    parser.add_argument('--checkpoint', default='./weights_kalapa/vgg_seq2seq_0.9395.pth', required=False, help='your checkpoint')

    args = parser.parse_args()
    config = Cfg.load_config_from_file(args.config)
    # config['vocab'] += '£€¥'
    config['cnn']['pretrained']=False
    trainer = Trainer(config)

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint, use_freeze=False)
        
    trainer.train()

if __name__ == '__main__':
    main()
