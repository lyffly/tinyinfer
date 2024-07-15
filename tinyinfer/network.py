import numpy as np
from .nodes import *


class Network():
    def __init__(self):
        super().__init__()
        self.nodes = {}
        self.edges = {}
        self.run_orders = []
        self.input_names = []
        self.output_names = []
        self.config = None
    
    def prepare(self, ins = {}):
        self.bind_all_edges()
        for key in ins.keys():
            in_tensor = ins[key]
            self.edges[key].shape = in_tensor.shape
            self.edges[key].dtype = in_tensor.dtype
            self.edges[key].tensor = in_tensor
        
        # 形状推导 tensor 进行绑定
        for nodename in self.run_orders:
            self.nodes[nodename].infer_shapes()

    def run(self, ins = {}):
        # 进行每一个node的推理
        for nodename in self.run_orders:
            if self.config.log_verbose:
                print("[run] node name: ", nodename)
            self.nodes[nodename].run()
        # 获取输出
        outs = {}
        for name in self.output_names:
            out_tensor = self.edges[name].tensor
            outs[name] = out_tensor
        print("[run] network run end ! \n", "*" * 80)

        return outs
        
    def bind_all_edges(self):
        for nodename in self.run_orders:
            self.nodes[nodename].bing_all_edges(self.edges)


