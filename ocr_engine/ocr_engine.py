import cv2
import time
from .dbnet import DBNet
from .crnn import CRNN

class TextBox():

    def __init__(self, xmin, ymin, xmax, ymax, text='', kls=0):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.text = text
        self.kls = kls

    @property
    def width(self):
        return self.xmax - self.xmin
    
    @property
    def height(self):
        return self.ymax - self.ymin

    @property
    def area(self):
        return (self.ymax - self.ymin) * (self.xmax - self.xmin)

    def gen_idcar(self):
        str_data = '{}, {}, {}, {}, {}, {}, {}, {}, {}'.format(
            self.xmin, self.ymin,
            self.xmin, self.ymax,
            self.xmax, self.ymax,
            self.xmax, self.ymin,
            self.kls
        )
        return str_data


class OCREngine():

    def __init__(
        self,
        dbnet_onnx,
        crnn_onnx
    ):
        self.dbnet = DBNet(
            model_path=dbnet_onnx,
            max_length=640,
            min_contour_size=5,
            binary_threshold=0.1,
            polygon_threshold=0.,
            max_candidates=1000,
            unclip_ratio=1.5
        )
        if crnn_onnx is not None:
            self.crnn = CRNN(
                model_path=crnn_onnx,
                height=32,
                width=200,
                vocabulary="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz| 0123456789ĂÂÊÔƠƯÁẮẤÉẾÍÓỐỚÚỨÝÀẰẦÈỀÌÒỒỜÙỪỲẢẲẨĐẺỂỈỎỔỞỦỬỶÃẴẪẼỄĨÕỖỠŨỮỸẠẶẬẸỆỊỌỘỢỤỰỴăâêôơưáắấéếíóốớúứýàằầèềìòồờùừỳảẳẩđẻểỉỏổởủửỷãẵẫẽễĩõỗỡũữỹạặậẹệịọộợụựỵ'*:,@.-(#%\")/~!^&_´+={}[]\;<>?※”$€£¥₫°²™ā–"
            )
        else:
            self.crnn = None

    def __call__(self, image):
        bboxes, scores = self.dbnet(image)
        textboxes = []
        for bbox in bboxes:
            textbox = TextBox(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            if self.crnn is not None:
                crop_image = image[
                    textbox.ymin:textbox.ymax,
                    textbox.xmin:textbox.xmax
                ]
                text, score = self.crnn(crop_image)
                textbox.text = text
            textboxes.append(textbox)

        return textboxes
    
    def predict_crnn(self, image):
        return self.crnn(image)
    
    def predict_db(self, image):
        bboxes, _ = self.dbnet(image)
        return bboxes

        #     bbox = [int(x) for x in bbox]
        #     xmin, ymin, xmax, ymax = bbox
        #     image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)