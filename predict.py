from PIL import Image
import os
import glob
import cv2
import tqdm
import pandas as pd
from tool.predictor import Predictor
from tool.config import Cfg

def main():
    config = Cfg.load_config_from_file('config.yml')
    config['cnn']['pretrained']=False
    config['weights'] = '/media/hwr/TK/vietocr_basecode/weights_kalapa/vgg_seq2seq_0.8948.pth'
    detector = Predictor(config)
    # img = Image.open("/media/hwr/TK/OCR/public_test/images/215/17.jpg")
    # s = detector.predict(img)[0]
    # print(s)
    paths = glob.glob("/media/hwr/TK/OCR/public_test/images/*/*")
    res = {"id" : [], "answer" : []}
    for path in tqdm.tqdm(paths):
        img = Image.open(path)                                          
        s = detector.predict(img)[0]
        relative_path = path.split("/")[-2] + "/" + path.split("/")[-1]
        res["id"].append(relative_path)
        res["answer"].append(s)
    df = pd.DataFrame(res)
    df.to_csv(os.path.join("/media/hwr/TK/OCR/public_test", "sample_submission.csv"), index=False)

if __name__ == '__main__':
    main()