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

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = True

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
    #images = torch.zeros([1,3,224,224], dtype=torch.float32, requires_grad=False)
    inputs = {"images" : img}
    network.prepare(inputs)
    # warup
    for i in range(5):
        results = network.run(inputs)
    cudart.cudaDeviceSynchronize()
    start = time.time()
    # forward 
    for i in range(20):
        results = network.run(inputs)
    cudart.cudaDeviceSynchronize()
    end = time.time()
    fps = 20.0/(end - start)
    results["output"].cpu()
    out_ytensor = results["output"]
    shape = out_ytensor.shape
    shape_len = 1
    for s in shape:
        shape_len *= s
    outdata = np.frombuffer(out_ytensor.memoryview(), np.float32, shape_len)
    outdata = outdata.reshape(shape)
    out_tensor = softmax(outdata, axis=1)
    print("out shape:", out_tensor.shape)
    print("detect confidence:", np.max(out_tensor[0]))
    print("max index:", np.argmax(out_tensor[0]))
    print("imagenet label:", get_imagenet_labels(np.argmax(out_tensor[0])))
    print("fps:", fps)
    
    assert np.argmax(out_tensor[0]) == 22

