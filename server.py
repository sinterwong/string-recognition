from flask import Flask, jsonify, request
import cv2
import numpy as np
from infer import OnnxInference
from gevent import pywsgi
import urllib

app = Flask(__name__)
app.secret_key = "sinter"
app.debug = True


@app.route("/inference", methods=["POST"])
def inference():
    path = request.json.get("img_path")
    if path.split(":")[0] in ["http", "https"]:
        with urllib.request.urlopen(path) as url:
            resp = url.read()
            frame = np.asarray(bytearray(resp), dtype="uint8")
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    else:
        frame = cv2.imread(path)
    ret = engine._single_image(frame)
    ret = "".join([engine.idx2class[c] for c in ret])
    result = {
        "result": ret,
        "status": 200,
        "msg": "success"
    }
    return jsonify(result)


if __name__ == "__main__":
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    engine = OnnxInference("work_dirs/best_0.970.onnx", (80, 192), chars)
    server = pywsgi.WSGIServer(('0.0.0.0', 9797), app)
    server.serve_forever()
