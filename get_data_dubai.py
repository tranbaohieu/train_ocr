import glob
import os
import shutil
import random

# DATA_DUBAI = '/data/workdir/hieutb1/text_recognition_document/DATA_DUBAI'
# os.makedirs(DATA_DUBAI, exist_ok=True)
# DATA_DUBAI_IMAGES = '/data/workdir/hieutb1/text_recognition_document/DATA_DUBAI/images'
# os.makedirs(DATA_DUBAI_IMAGES, exist_ok=True)
# LABEL_FILE = '/data/workdir/hieutb1/text_recognition_document/DATA_DUBAI/labels.txt'
# f = open(LABEL_FILE, 'w')
# count = 0

# data_dir_list = ['/data/Data/DATA/dubaipetroleum_invoices/dbplus_label/20220722', '/data/Data/DATA/dubaipetroleum_invoices/dbplus_label/20221003_1year_000_029', \
#                  '/data/Data/DATA/dubaipetroleum_invoices/dbplus_label/20221016_1year_030_059']
# for data_dir in data_dir_list:
#     label_path_list = glob.glob(os.path.join(data_dir, '*/data_line_images/*/labels.txt'))
#     for label_path in label_path_list:
#         lines = open(label_path, 'r').readlines()
#         for line in lines[:-1]:
#             name_img, label = line.strip().split(maxsplit=1)
#             img_path = os.path.join(os.path.dirname(label_path), name_img)
#             imgs_dir = os.path.join(DATA_DUBAI_IMAGES, str(count//1000))
#             os.makedirs(imgs_dir, exist_ok=True)
#             des_path = os.path.join(imgs_dir, str(count)+'.jpg')
#             shutil.copy(img_path, des_path)
#             f.write('{}\t{}\n'.format(des_path, label.strip()))
#             count += 1
#             if count % 1000 == 0:
#                 print(count)
    #         if count > 3000:
    #             break
    #     if count > 3000:
    #         break
    # if count > 3000:
    #     break        

data_dir = '/data/workdir/hieutb1/text_recognition_document/DATA_CMC'
lines = open(os.path.join(data_dir, 'labels.txt'), 'r').readlines()
random.shuffle(lines)
lines_test = lines[:5000]
lines_train = lines[5000:]
f_test = open('/data/workdir/hieutb1/text_recognition_document/DATA_CMC_VAL/labels.txt', 'w')
for line in lines[:5000]:
    f_test.write(line)
f_train = open('/data/workdir/hieutb1/text_recognition_document/DATA_CMC/labels_2.txt', 'w')
for line in lines[5000:]:
    f_train.write(line)
f_test.close()
f_train.close()