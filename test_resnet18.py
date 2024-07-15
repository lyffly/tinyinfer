import torch
import cv2

from tinyinfer.runtime import import_model, build_network
from tinyinfer.config import Config

if __name__ == "__main__":
    img_name = "data/eagle.jpg"
    img = cv2.imread(img_name,1)
    
    onnx_name = "data/resnet18.onnx"
    
    config = Config()
    config.log_verbose = True
    config.fp32 = True
    config.use_cpu = True
    
    model_data = import_model(onnx_name, config)
    network = build_network(model_data, config)
    images = torch.zeros([1,3,224,224], dtype=torch.float32, requires_grad=False)
    inputs = {"images" : images}
    network.prepare(inputs)
    for i in range(100):
        results = network.run(inputs)

