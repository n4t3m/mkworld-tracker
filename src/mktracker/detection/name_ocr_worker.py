"""Persistent PaddleOCR recognizer for result text and rank crops.

The app uses a lightweight environment. Run this worker with a separate Python
environment containing PaddlePaddle and the PaddleOCR source checkout.
Each input line is JSON with base64 PNG crops and an optional input width;
each reply starts with MKW_OCR:.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--paddleocr", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.paddleocr))

    import numpy as np
    import paddle
    import yaml
    from ppocr.data import create_operators, transform
    from ppocr.modeling.architectures import build_model
    from ppocr.postprocess import build_post_process

    paddle.set_device("cpu")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    postprocess = build_post_process(config["PostProcess"], config["Global"])
    config["Architecture"]["Head"]["out_channels"] = len(postprocess.character)
    model = build_model(config["Architecture"])
    model.set_state_dict(paddle.load(str(args.checkpoint)))
    model.eval()
    ops_by_width = {
        width: create_operators([
            {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
            {"RecResizeImg": {"image_shape": [3, 48, width]}},
            {"KeepKeys": {"keep_keys": ["image"]}},
        ], config["Global"])
        for width in (64, 320)
    }
    print("MKW_OCR:" + json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("shutdown"):
                print("MKW_OCR:" + json.dumps({"stopped": True}), flush=True)
                break
            width = int(request.get("width", 320))
            if width not in ops_by_width:
                raise ValueError("Unsupported OCR image width")
            images = [transform({"image": base64.b64decode(encoded)}, ops_by_width[width])[0]
                      for encoded in request["crops"]]
            with paddle.no_grad():
                decoded = postprocess(model(paddle.to_tensor(np.stack(images)))) if images else []
            response = {"results": [[name, float(conf)] for name, conf in decoded]}
        except Exception as error:
            response = {"error": str(error)}
        print("MKW_OCR:" + json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
