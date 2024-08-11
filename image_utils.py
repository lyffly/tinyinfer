import torch
from torchvision import transforms as T
import numpy as np
import cv2
from PIL import Image

def opencv_to_pillow(img) :
    img= Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    return img

def imagenet_preprocess(filename):
    input_image = Image.open(filename)
    preprocess = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

def yolox_preprocess(filename):
    img = cv2.imread(filename,1)
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2,0,1))
    input_batch = np.expand_dims(img, 0)
    
    return torch.from_numpy(input_batch)

def read_image(name, hw=(224,224)):
    img = cv2.imread(name,1)
    img = cv2.resize(img, hw, interpolation=cv2.INTER_NEAREST)
    return img


def get_imagenet_labels(index):
    import json
    labels = []
    with open('data/imagenet_labels.json') as f:
        labels = json.load(f)
    return labels[index]
