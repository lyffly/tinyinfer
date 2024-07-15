import torch
import numpy as np
import cv2

def imagenet_preprocess(img):
    pass


def read_image(name):
    img = cv2.imread(name,1)
    return img
