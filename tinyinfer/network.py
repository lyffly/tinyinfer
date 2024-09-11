import numpy as np
from .nodes import *
from .edges import *
import copy
import torch
from kernels import YTensor, DataType, DataLayout, TensorType
from .utils import numpy_dtype_2_ytensor_dtype, get_np_data_ptr
from cuda import cudart

class Network:
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
        self.stream = None
        self.ins_max_shape = None

    def prepare(self, ins:list, ins_max_shape=None):
        _, self.stream = cudart.cudaStreamCreate()
        self.bind_all_edges()
        for key in ins.keys():
            in_np = ins[key]
            self.edges[key].shape = in_np.shape
            self.edges[key].dtype = in_np.dtype
            ytensor = YTensor()
            ytensor.zeros(
                list(in_np.shape),
                numpy_dtype_2_ytensor_dtype(in_np.dtype),
                DataLayout.nchw,
            )
            ytensor.tensortype = TensorType.input
            ytensor.name = key
            self.edges[key].tensor = ytensor
        # 设置模型精度时，设置node的精度
        for nodename in self.run_orders:
            if self.config.fp32:
                self.nodes[nodename].set_op_precision("float32")
            if self.config.fp16:
                self.nodes[nodename].set_op_precision("float16")
            elif self.config.int8:
                print("[Error] not support int8 yet !!!")
                raise IOError
                self.nodes[nodename].set_op_precision("int8")

        # 设置模型精度时，在输入插入转换节点
        if self.config.fp16:
            for io_name, io_edge in copy.copy(self.edges).items():
                if io_edge.type == "input":
                    for node_name, _ in copy.copy(self.nodes).items():
                        if io_name in self.nodes[node_name].input_names:
                            to_add_edge = Edge()
                            to_add_edge.type = "variable"
                            to_add_edge.name = "Cast_out_" + str(len(self.edges))
                            self.edges[to_add_edge.name] = to_add_edge
                            to_add_node = CastNode("float32", "float16")
                            to_add_node.input_names = [io_name]
                            to_add_node.output_names = [to_add_edge.name]
                            to_add_node.name = "Cast_" + str(len(self.nodes))
                            self.nodes[to_add_node.name] = to_add_node

                            name_idx = self.nodes[node_name].input_names.index(io_name)
                            self.nodes[node_name].input_names[
                                name_idx
                            ] = to_add_edge.name
                            run_idx = self.run_orders.index(node_name)
                            self.run_orders.insert(run_idx, to_add_node.name)
                if io_edge.type == "output":
                    for node_name, _ in copy.copy(self.nodes).items():
                        if io_name in self.nodes[node_name].output_names:
                            to_add_edge = Edge()
                            to_add_edge.type = "variable"
                            to_add_edge.name = "Cast_out_" + str(len(self.edges))
                            self.edges[to_add_edge.name] = to_add_edge
                            to_add_node = CastNode("float16", "float32")
                            to_add_node.input_names = [to_add_edge.name]
                            to_add_node.output_names = [io_name]
                            to_add_node.name = "Cast_" + str(len(self.nodes))
                            self.nodes[to_add_node.name] = to_add_node

                            name_idx = self.nodes[node_name].output_names.index(io_name)
                            self.nodes[node_name].output_names[
                                name_idx
                            ] = to_add_edge.name
                            self.run_orders.append(to_add_node.name)
        self.bind_all_edges()
        # 设置最大形状
        self.ins_max_shape = ins_max_shape
        if self.ins_max_shape is not None:
            for key in ins.keys():
                self.edges[key].shape = self.ins_max_shape[key]
                self.edges[key].max_shape = self.ins_max_shape[key]
                self.edges[key].tensor.max_shape = self.ins_max_shape[key]
            for nodename in self.run_orders:
                self.nodes[nodename].set_op_shapes()
                self.nodes[nodename].set_op_max_shapes()
                if self.config.log_verbose:
                    print("[infer shape] node name: ", nodename)
                    out_edge = self.edges[self.nodes[nodename].output_names[0]]
                    print("     ---> out max shape: ", out_edge.shape)
        # 形状推导 tensor 进行绑定
        for key in ins.keys():
            in_np = ins[key]
            self.edges[key].shape = in_np.shape
        for nodename in self.run_orders:
            self.nodes[nodename].set_op_shapes()
            self.nodes[nodename].setup_op_out_edges()
            if self.config.log_verbose:
                print("[infer shape] node name: ", nodename)
                out_edge = self.edges[self.nodes[nodename].output_names[0]]
                print("     ---> out shape: ", out_edge.shape, out_edge.dtype)
        # move edge data to gpu
        for key in self.edges.keys():
            if self.edges[key].type == "input" and self.config.use_gpu:
                self.edges[key].tensor.cuda()
            elif self.edges[key].type == "output" and self.config.use_gpu:
                self.edges[key].tensor.cuda()
            elif self.config.use_gpu:
                self.edges[key].tensor.cuda()

    def run(self, ins={}):
        # 把输入数据转到gpu
        for key in self.edges.keys():
            if self.edges[key].type == "input" and self.config.use_gpu:
                in_np = ins[key]
                self.edges[key].tensor.copy_numpy_data(get_np_data_ptr(in_np))
                self.edges[key].shape = in_np.shape
        
        # 每次开始前获取最新的形状
        for nodename in self.run_orders:
            self.nodes[nodename].set_op_shapes()
            if self.config.log_verbose:
                out_edge = self.nodes[nodename].all_edges[self.nodes[nodename].output_names[0]]
                print("[set output shape] node name: {0}  -> out0 shape: {1}".format(nodename, out_edge.shape))
        
        # 进行每一个node的推理
        for nodename in self.run_orders:
            if self.config.log_verbose:
                print("[run] node name: ", nodename)
            self.nodes[nodename].run(self.stream)
        # 获取输出，输出数转到cpu
        outs = {}
        for name in self.output_names:
            # if self.config.use_gpu:
            #     self.edges[name].tensor.cpu()
            outs[name] = self.edges[name].tensor

        if self.config.log_verbose:
            print("[run] network run end ! \n\n", "*" * 80)

        return outs

    def bind_all_edges(self):
        for nodename in self.run_orders:
            self.nodes[nodename].bind_all_edges(self.edges)
