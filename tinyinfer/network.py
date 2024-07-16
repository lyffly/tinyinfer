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
        self.nodes_num = 0
        self.edges_num = 0
    
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
            if self.config.log_verbose:
                print("[infer shape] node name: ", nodename)
                out_edge = self.edges[self.nodes[nodename].output_names[0]]
                print("     -> ", out_edge.shape)
        # move edge data to gpu
        for key in self.edges.keys():
            if self.edges[key].type == "input":
                pass
            elif self.edges[key].type == "output":
                pass
            elif self.config.use_gpu:
                self.edges[key].tensor = self.edges[key].tensor.to(self.config.gpu_device)
        

    def run(self, ins = {}):
        # 把输入输出数据转到gpu
        for key in self.edges.keys():
            if self.edges[key].type == "input" and self.config.use_gpu:
                self.edges[key].tensor = self.edges[key].tensor.to(self.config.gpu_device)
            elif self.edges[key].type == "output" and self.config.use_gpu:
                self.edges[key].tensor = self.edges[key].tensor.to(self.config.gpu_device)

        # 进行每一个node的推理
        for nodename in self.run_orders:
            if self.config.log_verbose:
                print("[run] node name: ", nodename)
            self.nodes[nodename].run()
        # 获取输出，输出数转到cpu
        outs = {}
        for name in self.output_names:
            out_tensor = self.edges[name].tensor
            if self.config.use_gpu:
                out_tensor = out_tensor.cpu()
            outs[name] = out_tensor
                
        if self.config.log_verbose:
            print("[run] network run end ! \n", "*" * 80)

        return outs

    def bind_all_edges(self):
        for nodename in self.run_orders:
            self.nodes[nodename].bing_all_edges(self.edges)


