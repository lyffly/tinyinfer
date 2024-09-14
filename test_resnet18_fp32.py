import torch
import cv2
import numpy as np
from scipy.special import softmax

np.set_printoptions(precision=3)

from tinyinfer.runtime import import_model, build_network
from tinyinfer.config import Config
from tinyinfer.utils import ytensor_2_numpy
import tinyinfer
from image_utils import *
from cuda import cudart
import time

if __name__ == "__main__":
    tinyinfer.get_gpu_info()
    img_name = "data/eagle.jpg"
    img = imagenet_preprocess(img_name)

    onnx_name = "data/resnet18.onnx"

    config = Config()
    config.log_verbose = False
    config.fp32 = True
    config.fp16 = False
    config.use_cpu = False
    config.use_gpu = True

    model_data = import_model(onnx_name, config)
    network = build_network(model_data, config)

    inputs_zeros = {"images": np.zeros((1,3,224,224)).astype(np.float32)}
    inputs_max_shape = {"images": [1,3,224,224]}
    network.prepare(inputs_zeros, inputs_max_shape)
    
    inputs = {"images": img.numpy()}
    # warup
    for i in range(5):
        results = network.run(inputs)
    cudart.cudaDeviceSynchronize()
    start = time.time()
    # forward
    loops = 50
    for i in range(loops):
        results = network.run(inputs)
    cudart.cudaDeviceSynchronize()
    end = time.time()
    fps = float(loops) / (end - start)
    out_ytensor = results["output"]
    # out_ytensor.print(10)
    outdata = ytensor_2_numpy(out_ytensor)
    out_tensor = softmax(outdata, axis=1)
    # print(out_tensor[0][:10])
    print("out shape:", out_tensor.shape)
    print("detect confidence:", np.max(out_tensor[0]))
    print("max index:", np.argmax(out_tensor[0]))
    print("imagenet label:", get_imagenet_labels(np.argmax(out_tensor[0])))
    print("fps:", fps)
    del network

    assert np.argmax(out_tensor[0]) == 22
