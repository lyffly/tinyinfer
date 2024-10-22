import torch
import cv2
import numpy as np
from scipy.special import softmax

np.set_printoptions(precision=3)

from tinyinfer.runtime import import_model, build_network
from tinyinfer.config import Config
import tinyinfer
from image_utils import *
from cuda import cudart
import time

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    tinyinfer.get_gpu_info()
    img_name = "data/dog.jpg"
    img = yolox_preprocess(img_name)

    onnx_name = "data/yolox_s.onnx"

    config = Config()
    config.log_verbose = True
    config.fp32 = False
    config.fp16 = True
    config.use_cpu = False
    config.use_gpu = True

    model_data = import_model(onnx_name, config)
    network = build_network(model_data, config)
    # img = torch.zeros([1,3,640,640], dtype=torch.float32, requires_grad=False)
    inputs = {"images": img}
    network.prepare(inputs)
    # warup
    for i in range(5):
        results = network.run(inputs)
    cudart.cudaDeviceSynchronize()
    start = time.time()
    # forward
    for i in range(10):
        results = network.run(inputs)
    cudart.cudaDeviceSynchronize()
    end = time.time()
    fps = 10.0 / (end - start)
    out_tensor = results["output"].cpu().numpy()

    print("out shape:", out_tensor.shape)
    # print("detect confidence:", np.max(out_tensor[0]))
    # print("max index:", np.argmax(out_tensor[0]))
    # print("imagenet label:", get_imagenet_labels(np.argmax(out_tensor[0])))
    # print("fps:", fps)

    # assert np.argmax(out_tensor[0]) == 22
