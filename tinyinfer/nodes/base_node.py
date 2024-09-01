from ..params import *
import math
import torch
import numpy as np
import torch.nn.functional as F
from cuda import cudart
import kernels
from copy import deepcopy


class Node:
    def __init__(self):
        self.name = None
        self.type = None
        self.params = None
        self.input_names = None
        self.output_names = None
        self.input_shapes = None
        self.output_shapes = None
        self.input_dtypes = None
        self.output_dtypes = None
        self.input_layouts = None
        self.output_layouts = None
        self.all_edges = None
        self.network_precision = "float32"

    def bind_all_edges(self, edges):
        self.all_edges = edges

    def run(self, stream):
        print("node run")
        pass

    def print(self):
        print("\nnode name:", self.name)
        print("     type:", self.type)
