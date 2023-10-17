import cv2
import numpy as np
from shapely.geometry import Polygon
import pyclipper

from .onnx_engine import ONNXEngine


class DBNet():

    def __init__(
        self,
        model_path,
        max_length,
        min_contour_size,
        binary_threshold,
        polygon_threshold,
        max_candidates,
        unclip_ratio,
    ):
        self.engine = ONNXEngine(model_path)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.max_length = max_length
        self.min_contour_size = min_contour_size
        self.binary_threshold = binary_threshold
        self.polygon_threshold = polygon_threshold
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    def __call__(self, image):
        input = self.preprocess(image)
        output = self.engine(input)
        bboxes, scores = self.postprocess(output)

        return bboxes, scores

    def preprocess(self, image):
        self.ori_h, self.ori_w = image.shape[:2]

        self.scale = min(self.max_length/min(self.ori_h, self.ori_w), 1)
        w = int(self.ori_w*self.scale/32)*32
        h = int(self.ori_h*self.scale/32)*32
        
        image = cv2.resize(image, (w, h))

        image = image.astype(np.float32) / 255.0
        image -= self.mean
        # image /= self.std

        input = np.transpose(image, (2, 0, 1))
        input = np.expand_dims(input, axis=0)
        return input

    def postprocess(self, pred):
        mask = pred[0].squeeze()
        mask = cv2.resize(mask, (self.ori_w, self.ori_h))
        _, bitmap = cv2.threshold(
            mask, self.binary_threshold, 255, cv2.THRESH_BINARY)
        height, width = bitmap.shape
        # bitmap = 255 - bitmap
        contours, _ = cv2.findContours(
            (bitmap).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)
        bboxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_contour_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(mask, points.reshape(-1, 2))
            if self.polygon_threshold > score or score < 0.01:
                continue
            box = points
            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes2(box)
            if sside < self.min_contour_size + 2:
                continue
            box = np.array(box)
            box[:, 0] = np.clip(box[:, 0] / width, 0, 1)*self.ori_w
            box[:, 1] = np.clip(box[:, 1] / height, 0, 1)*self.ori_h
            bboxes.append([box[0][0], box[0][1], box[2][0], box[2][1]])
            scores.append(score)
        bboxes = np.array(bboxes).astype(np.int)
        scores = np.array(scores).astype(np.float)

        return bboxes, scores

    def unclip(self, box):
        poly = Polygon(box)
        distance = poly.area * self.unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2],
               points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def get_mini_boxes2(self, contour):
        x, y, w, h = cv2.boundingRect(contour)
        return np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]]).astype(np.float32), min(w, h)

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        if xmin == xmax or ymin == ymax:
            return 0
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

