import argparse
from PIL import Image
import json
import numpy as np
import os
import glob
import cv2
import pandas as pd
import tqdm
from ocr_engine import OCREngine
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# orderprojection
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

def get_data_from_json(file, img_dir):
    gt = json.load(open(file=file))
    wordtxtes = []
    boxes = []
    for key in gt.keys():
        words_of_img = gt[key]
        img = cv2.imread(os.path.join(img_dir, key))
        for word in words_of_img:
            pts = [[pt['x'], pt['y']] for pt in word['boundingBox']['vertices']]
            pts = np.array(pts).reshape((-1, 2))
            wordimg = four_point_transform(pts=pts) 
            wordtext = ''
            for char in word['symbols']:
                wordtext += char['text']
            wordtxtes.append(wordtext)
    return boxes, wordtxtes

def main():
    config = Cfg.load_config_from_file('/data/workdir/hieutb1/text_recognition_document/vietocr/config.yml')
    config['weights'] = '/data/workdir/hieutb1/text_recognition_document/vietocr/weights/vgg_seq2seq_0.9058.pth'
    config['vocab'] += ' '
    config['cnn']['pretrained']=False
    config['device'] = 'cuda:0'
    detector = Predictor(config)
    ocr_engine = OCREngine(dbnet_onnx = '/data/checkpoints/dbnet/12072022_res50dbpp.onnx',
                        crnn_onnx = '/data/workdir/hieutb1/text_detect_document/generate_datasets/crnn/model.onnx')

    img_dir = '/data/workdir/hieutb1/text_detect_document/data_test/images'
    gt = json.load(open('/data/workdir/hieutb1/text_detect_document/data_test/sample.json'))
    names = []
    acces = []
    acces_idp = []
    num_correct = 0
    num_of_all = 0
    num_correct_idp = 0
    # num_of_all = 0
    for key in tqdm.tqdm(gt.keys()):
        names.append(str(key))
        words_of_img = gt[key]
        img = cv2.imread(os.path.join(img_dir, key))
        num_word_page = 0
        num_correct_page = 0
        num_correct_page_idp = 0
        for word in words_of_img:
            pts = [[pt['x'], pt['y']] for pt in word['boundingBox']['vertices']]
            pts = np.array(pts).reshape((-1, 2))
            wordimg = four_point_transform(pts=pts, image=img)
            wordimg_pil = Image.fromarray(wordimg)
            s = detector.predict(wordimg_pil)
            # s_idp, _ = ocr_engine.predict_crnn(wordimg)
            s_idp = ''
            wordtext = ''
            for char in word['symbols']:
                wordtext += char['text']
            if s == wordtext:
                num_correct += 1
                num_correct_page += 1
            if s_idp == wordtext:
                num_correct_idp += 1
                num_correct_page_idp += 1
            num_of_all += 1
            num_word_page += 1
        acces.append(num_correct_page/num_word_page)
        acces_idp.append(num_correct_page_idp/num_word_page)

        # print('{} : {}'.format(str(key), str(num_correct_page/num_word_page)))
        # print('{} : {}'.format(str(key), str(num_correct_page_idp/num_word_page)))
    d = {'name' : names, 'acc_new_ocr': acces, 'acc_old_ocr' : acces_idp}
    df = pd.DataFrame(data=d)
    df.to_csv('/data/workdir/hieutb1/text_detect_document/data_test/compare_crnn.csv')
    print('acc_all_new : ',num_correct/num_of_all)
    print('acc_all_new : ',num_correct_idp/num_of_all)
    

if __name__ == '__main__':
    main()
