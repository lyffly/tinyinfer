import torch
import cv2

from tinyinfer.runtime import import_model, build_network
from tinyinfer.config import Config
from image_utils import *

if __name__ == "__main__":
    img_name = "data/eagle.jpg"
    img = read_image(img_name)
    img = imagenet_preprocess(img)
    img = torch.unsqueeze(img, 0)
    
    onnx_name = "data/resnet18.onnx"
    
    config = Config()
    config.log_verbose = True
    config.fp32 = True
    config.use_cpu = True
    
    model_data = import_model(onnx_name, config)
    network = build_network(model_data, config)
    #images = torch.zeros([1,3,224,224], dtype=torch.float32, requires_grad=False)
    inputs = {"images" : img}
    network.prepare(inputs)
    for i in range(100):
        results = network.run(inputs)
    
    out_tensor = results["output"].numpy()
    print(out_tensor.shape)
    print(out_tensor)
