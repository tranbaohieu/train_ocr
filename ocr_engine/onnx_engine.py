import onnxruntime


class ONNXEngine():

    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

    def __call__(self, input):
        ort_input = {self.session.get_inputs()[0].name: input}
        output = self.session.run(None, ort_input)[0]
        return output
