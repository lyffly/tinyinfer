import cv2
import numpy as np
from scipy.special import softmax
np.set_printoptions(precision=3)

import onnxruntime as ort
from image_utils import *
from cuda import cudart
import time


if __name__ == "__main__":
    img_name = "data/eagle.jpg"
    img = imagenet_preprocess(img_name)
    
    onnx_name = "data/resnet18.onnx"
    
    session = ort.InferenceSession(onnx_name, providers=['CUDAExecutionProvider'])
    
    #images = torch.zeros([1,3,224,224], dtype=torch.float32, requires_grad=False)
    inputs = {"images" : img.cpu().numpy()}
    # warup
    for i in range(5):
        ort_outputs = session.run([], inputs)[0]
    
    cudart.cudaDeviceSynchronize()
    start = time.time()
    # forward 
    for i in range(20):
        ort_outputs = session.run([], inputs)[0]
    cudart.cudaDeviceSynchronize()
    end = time.time()
    fps = 20.0/(end - start)
    out_tensor = ort_outputs
    out_tensor = softmax(out_tensor, axis=1)
    print("out shape:", out_tensor.shape)
    print("detect confidence:", np.max(out_tensor[0]))
    print("max index:", np.argmax(out_tensor[0]))
    print("imagenet label:", get_imagenet_labels(np.argmax(out_tensor[0])))
    print("fps:", fps)
    
    assert np.argmax(out_tensor[0]) == 22

