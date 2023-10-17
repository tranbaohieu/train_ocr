import numpy as np
import cv2
from .onnx_engine import ONNXEngine


class CRNN():

    def __init__(
        self,
        model_path,
        vocabulary,
        height,
        width
    ):
        self.engine = ONNXEngine(model_path)
        self.vocabulary = vocabulary
        self.height = height
        self.width = width
        self.mean = 127.5
        self.std = 255

    def __call__(self, image):
        input = self.preprocess(image)
        output = self.engine(input=input)
        texts, scores = self.postprocess(output)

        return texts[0], scores[0]

    def preprocess(self, image):
        image = cv2.resize(image, (self.width, self.height))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = (image.astype(np.float32) - self.mean) / self.std
        input = np.expand_dims(image, axis=[0, 1])
        return input

    def postprocess(self, inputs):
        batch = inputs.shape[0]
        list_text = []
        list_confidences = []
        for i in range(batch):
            text_index = inputs[i, : ,:]
            confidences = self.softmax(text_index, axis=1)
            text_index = np.argmax(confidences, axis=1)
            text, conf = self.decode(text_index, confidences)
            list_text.append(text)
            list_confidences.append(conf)
        return list_text, list_confidences


    def decode(self, text_index, confidences):
        list_char = []
        list_confidences = []
        for i in range(len(text_index)):
            if text_index[i] != 0 and (not (i > 0 and text_index[i - 1] == text_index[i])):  # removing repeated characters and blank.
                list_char.append(self.vocabulary[text_index[i] - 1])
                list_confidences.append(float(confidences[i, text_index[i]]))
        text = ''.join(list_char)
        return text, list_confidences


    def softmax(self, X, theta=1.0, axis=None):
        # make X at least 2d
        y = np.atleast_2d(X)
        # find axis
        if axis is None:
            axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
        # multiply y against the theta parameter,
        y = y * float(theta)
        # subtract the max for numerical stability
        y = y - np.expand_dims(np.max(y, axis=axis), axis)
        # exponentiate y
        y = np.exp(y)
        # take the sum along the specified axis
        ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
        # finally: divide elementwise
        p = y / ax_sum
        # flatten if X was 1D
        if len(X.shape) == 1: p = p.flatten()
        return p
