import simplejson as json
import numpy as np
import os
from copy import deepcopy

from .nodes import *
from .edges import *
from .params import *
from .network import *
from .onnx_to_bin import convert_to_bin

# bytes_to_int
def bytes_to_int(bytes) :
    return int.from_bytes(bytes, byteorder='little')


def bytes_to_str(bytes):
    return bytes.decode("utf-8")

# 
def str_to_bytes(str):
    return bytes(str, 'utf-8')


# read int from file
def file_read_int(f, num=8):
    bytes = f.read(num)
    if not bytes:
        return None
    return bytes_to_int(bytes)

def file_read_bytes(f, num=8):
    bytes = f.read(num)
    if not bytes:
        return None
    return bytes


# read str from file
def file_read_str(f, length=1):
    bytes = f.read(length)
    if not bytes:
        return None
    return bytes_to_str(bytes)

def import_model(in_name, config):
    name = in_name
    if in_name.endswith(".onnx") :
        name = convert_to_bin(in_name)
    elif in_name.endswith(".bin") :
        name = in_name
    else :
        raise FileNotFoundError
    with open(name, "rb") as f:
        length = file_read_int(f)
        #print(length)
        bin_data = f.read(length)
        bin_data = bytes_to_str(bin_data)
        #bin_data = str(bin_data)
        net_data = json.loads(bin_data)
        is_end = False
        
        net_data["tensor_data"] = {}
        while (not is_end):
            str_len = file_read_int(f)
            if not str_len :
                is_end = True
                break
            tensorname = file_read_str(f, str_len)
            data_len = file_read_int(f)
            tensordata = file_read_bytes(f, data_len)
            
            net_data["tensor_data"][tensorname] = tensordata
    return net_data


def build_network(bin_data, config):
    edges = bin_data["edges"]
    nodes = bin_data["nodes"]
    tenor_data = bin_data["tensor_data"]
    network = Network()
    network.config = config
    network.input_names = bin_data["inputs"]
    network.output_names = bin_data["outputs"]
    # all nodes
    for key in nodes.keys():
        value = nodes[key]
        
        attrbiute = value["attrbiute"]
        if value["type"] == "Conv":
            node = ConvNode()
            node.input_names = value["inputs"]
            node.output_names = value["outputs"]
            node.name = value["name"]
            node.type = value["type"]
            params = ConvParams()
            params.strides = attrbiute["strides"]
            params.dilations = attrbiute["dilations"]
            params.kernel_shape = attrbiute["kernel_shape"]
            params.pads = attrbiute["pads"]
            params.group = attrbiute["group"]
            node.params = params
        elif value["type"] == "Relu":
            node = ActivationNode()
            node.input_names = value["inputs"]
            node.output_names = value["outputs"]
            node.name = value["name"]
            node.type = value["type"]
        elif value["type"] == "Sigmoid":
            node = ActivationNode()
            node.input_names = value["inputs"]
            node.output_names = value["outputs"]
            node.name = value["name"]
            node.type = value["type"]
        elif value["type"] == "Add":
            node = ElementwiseNode()
            node.input_names = value["inputs"]
            node.output_names = value["outputs"]
            node.name = value["name"]
            node.type = value["type"]
        elif value["type"] == "Div":
            node = ElementwiseNode()
            node.input_names = value["inputs"]
            node.output_names = value["outputs"]
            node.name = value["name"]
            node.type = value["type"]
        elif value["type"] == "Sub":
            node = ElementwiseNode()
            node.input_names = value["inputs"]
            node.output_names = value["outputs"]
            node.name = value["name"]
            node.type = value["type"]
        elif value["type"] == "Mul":
            node = ElementwiseNode()
            node.input_names = value["inputs"]
            node.output_names = value["outputs"]
            node.name = value["name"]
            node.type = value["type"]
        elif value["type"] == "Gemm":
            node = GemmNode()
            node.input_names = value["inputs"]
            node.output_names = value["outputs"]
            node.name = value["name"]
            node.type = value["type"]
            params = GemmParams()
            params.alpha = attrbiute["alpha"]
            params.beta = attrbiute["beta"]
            params.transB = attrbiute["transB"]
            node.params = params
        elif value["type"] == "Flatten":
            node = FlattenNode()
            node.input_names = value["inputs"]
            node.output_names = value["outputs"]
            node.name = value["name"]
            node.type = value["type"]
            params = FlattenParams()
            params.axis = attrbiute["axis"]
            node.params = params
        elif value["type"] == "Concat":
            node = ConcatNode()
            node.input_names = value["inputs"]
            node.output_names = value["outputs"]
            node.name = value["name"]
            node.type = value["type"]
            params = ConcatParams()
            params.axis = attrbiute["axis"]
            node.params = params
        elif value["type"] == "MaxPool":
            node = PoolNode()
            node.input_names = value["inputs"]
            node.output_names = value["outputs"]
            node.name = value["name"]
            node.type = value["type"]
            params = PoolParams()
            params.strides = attrbiute["strides"]
            params.kernel_shape = attrbiute["kernel_shape"]
            params.pads = attrbiute["pads"]
            params.ceil_mode = attrbiute["ceil_mode"]
            node.params = params
        elif value["type"] == "GlobalAveragePool":
            node = PoolNode()
            node.input_names = value["inputs"]
            node.output_names = value["outputs"]
            node.name = value["name"]
            node.type = value["type"]
        else:
            print("node type not impl: ", node.type, ", name :", node.name)
        network.nodes[node.name] = node
        network.run_orders.append(node.name)
    
    # all oonstant edges
    for key in edges:
        value = edges[key]
        edge = Edge()
        edge.name = value["name"]
        edge.shape = value["shape"]
        edge.dtype = value["dtype"]
        edge.is_constant = True
        edge.type = "constant" # constant input output variable
        # edge.tensor = torch.zeros(edge.shape, dtype=torch.float, requires_grad=False)
        np_data = deepcopy(np.frombuffer(tenor_data[edge.name], dtype=np.float32))
        edge.tensor = torch.from_numpy(np_data).reshape(edge.shape)
        edge.tensor.requires_grad_(False)
        network.edges[edge.name] = edge
    
    # networks in out edges
    for name in network.input_names:
        edge = Edge()
        edge.name = name
        edge.type = "input" # constant input output variable
        network.edges[edge.name] = edge
    for name in network.output_names:
        edge = Edge()
        edge.name = name
        edge.type = "output" # constant input output variable
        network.edges[edge.name] = edge
    
    # node's in out edges
    for key in network.nodes.keys():
        node = network.nodes[key]
        in_names = node.input_names
        out_names = node.output_names
        for name in in_names:
            if not name in network.edges.keys():
                edge = Edge()
                edge.name = name
                edge.type = "variable" # constant input output variable
                network.edges[edge.name] = edge
        for name in out_names:
            if not name in network.edges.keys():
                edge = Edge()
                edge.name = name
                edge.type = "variable" # constant input output variable
                network.edges[edge.name] = edge
    network.edges_num = len(network.edges)
    network.nodes_num = len(network.nodes)
    return network

