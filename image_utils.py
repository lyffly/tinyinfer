import torch
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image

def opencv_to_pillow(img) :
    img= Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    return img

def imagenet_preprocess(img):
    normalize = transforms.Normalize([0.22430152, 0.1806223,  0.14452167],
                                     [0.0927146,  0.07297731, 0.06149867])
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    img = transform(opencv_to_pillow(img))
    return img

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
