from flask import Flask, jsonify, request
import cv2
import numpy as np
from infer import OnnxInference
from gevent import pywsgi
import urllib
import argparse
import base64
from PIL import Image
import io

app = Flask(__name__)
app.secret_key = "sinter"
app.debug = True


@app.route("/inference", methods=["POST"])
def inference():
    result = {
        "status": 0,
        "msg": "success"
    }
    path = request.json.get("img_path")
    image_base64 = request.json.get("image_base64")

    frame = None
    if path:
        if path.split(":")[0] in ["http", "https"]:
            with urllib.request.urlopen(path) as url:
                resp = url.read()
                frame = np.asarray(bytearray(resp), dtype="uint8")
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        else:
            frame = cv2.imread(path)

    elif image_base64:
        data = base64.b64decode(image_base64)
        data = np.fromstring(data, np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is not None:
        ret = engine._single_image(frame)
        ret = "".join([engine.idx2class[c] for c in ret])
        result["result"] = ret
    else:
        result["msg"] = "请输入正确的图片"
        result["status"] = -5
        result["result"] = 0
    return jsonify(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="work_dirs/best_0.970.onnx", help="onnx model path")
    parser.add_argument("--ip", type=str,
                        default="0.0.0.0", help="onnx model path")
    parser.add_argument("--port", type=int,
                        default=22999, help="watching port")
    
    opt = parser.parse_args()
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    engine = OnnxInference(opt.model_path, (32, 128), chars)
    server = pywsgi.WSGIServer((opt.ip, opt.port), app)
    print(server.address)
    server.serve_forever()
