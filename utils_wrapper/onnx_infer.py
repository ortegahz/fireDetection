import sys
from pathlib import Path

import cv2

BaseDir = str(Path(__file__).resolve().parent.parent)
sys.path.append(BaseDir)

import onnxruntime
import numpy as np


class ONNXModel():
    def __init__(self, weights, providers=["CPUExecutionProvider"]):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(str(weights), providers=providers)
        self.input_names, self.input_shapes = self.get_input_name(self.onnx_session)

        self.output_names = self.get_output_name(self.onnx_session)

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_names = []
        for node in onnx_session.get_outputs():
            output_names.append(node.name)
        return output_names

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_names = []
        input_shapes = []
        for node in onnx_session.get_inputs():
            input_names.append(node.name)
            input_shapes.append(node.shape)
        return input_names, input_shapes

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def forward(self, image_numpy):
        '''
        # image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        input_feed = self.get_input_feed(self.input_names, image_numpy)
        res = self.onnx_session.run(self.output_names, input_feed=input_feed)

        return res

    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    # fake_img = np.random.random((1, 3, 736, 960)).astype(np.float32)
    fake_img = cv2.imread('/home/manu/tmp/60m4.png')
    fake_img = cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB)
    fake_img = np.transpose(fake_img, (2, 0, 1))
    fake_img = fake_img.astype(np.float32)[np.newaxis, :, :, :]
    fake_img = fake_img / 255.
    weights = Path("/home/manu/tmp/mm9.onnx")
    model = ONNXModel(weights)
    res = model.forward(fake_img)
    print(len(res))
    for save_i, feature in enumerate(res):
        print(feature.shape)
        save_output = feature.flatten()
        np.savetxt('/home/manu/tmp/onnx_yolo11_output_%s.txt' % save_i, save_output, fmt="%f", delimiter="\n")
