import cv2
import numpy as np
import onnxruntime
import os
from imutils import paths
import tqdm


class OnnxInference():
    def __init__(self, model_path, input_size, classes):
        self.idx2class = dict(enumerate(classes))
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_size = input_size
        self._load_model(model_path)

    def _load_model(self, weights):
        self.session = onnxruntime.InferenceSession(weights)
        self.input_name = self._get_input_name(self.session)
        self.output_name = self._get_output_name(self.session)

    def _get_output_name(self, session):
        """
        output_name = session.get_outputs()[0].name
        :param session:
        :return:
        """
        output_name = []
        for node in session.get_outputs():
            output_name.append(node.name)
        return output_name

    def _get_input_name(self, session):
        """
        input_name = session.get_inputs()[0].name
        :param session:
        :return:
        """
        input_name = []
        for node in session.get_inputs():
            input_name.append(node.name)
        return input_name

    def _get_input_feed(self, input_name, image_numpy):
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

    def _preprocess(self, frame):
        frame = cv2.resize(
            frame, (self.input_size[1], self.input_size[0])).astype(np.float32)
        frame /= 255.0
        return np.expand_dims(frame.transpose([2, 0, 1]), axis=0)

    def _single_image(self, frame):
        data = self._preprocess(frame)

        # forward_start = cv2.getTickCount()
        input_feed = self._get_input_feed(self.input_name, data)
        out = self.session.run(self.output_name, input_feed=input_feed)[0]
        ret = []
        for o in out:
            ret.append(int(np.argmax(o, axis=1)))
        # forward_end = cv2.getTickCount()
        # print("推理耗时：{}s".format((forward_end - forward_start) / cv2.getTickFrequency()))
        return ret

    def single_image(self, im_path, out_root):
        if not os.path.exists(out_root):
            os.makedirs(out_root)
        image = cv2.imread(im_path)
        pred = self._single_image(image[:, :, ::-1])  # bgr to rgb

        ret = "".join([self.idx2class[c] for c in pred])
        if not os.path.exists(out_root):
            os.makedirs(out_root)
        cv2.imwrite(os.path.join(out_root, ret + "_" +
                    os.path.basename(im_path)), image)

    def multi_images(self, im_root, out_root):
        im_paths = list(paths.list_images(im_root))
        start = cv2.getTickCount()
        [self.single_image(p, out_root)
         for p in tqdm.tqdm(im_paths, total=len(im_paths))]
        end = cv2.getTickCount()
        print("avg cost {} s".format((end - start) /
              cv2.getTickFrequency() / len(im_paths)))
        print("total cost {} s".format((end - start) / cv2.getTickFrequency()))


def main():
    ####### onnxruntime #######
    model_paths = "work_dirs/best_0.970.onnx"
    im_root = "datasets/val"
    out_root = "work_dirs/test_result"
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    engine = OnnxInference(model_paths, (80, 192), chars)
    engine.multi_images(im_root, out_root)


if __name__ == "__main__":
    main()
