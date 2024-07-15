import onnx
import simplejson as json

from onnx import numpy_helper
from onnx import  AttributeProto, TensorProto, GraphProto
import numpy as np

# int_to_bytes
def int_to_bytes(num) :
    return num.to_bytes(8,"little")

# bytes_to_int
def bytes_to_int(bytes) :
    return int.from_bytes(bytes, byteorder='little')

# 
def str_to_bytes(str):
    return bytes(str, 'utf-8')

# read int from file
def file_read_int(f, num=8):
    bytes = f.read(num)
    return bytes_to_int(bytes)

# write int to file
def file_write_int(f, num):
    f.write(int_to_bytes(num))

# write bytes to file
def file_write_bytes(f, bytes):
    f.write(bytes)

# write str to file
def file_write_str_with_len(f, str):
    str_bytes = bytes(str, 'utf-8')
    length = len(str_bytes)
    file_write_int(f, length)
    file_write_bytes(f, str_bytes)

# write bytes to file
def file_write_bytes_with_len(f, bytes):
    length = len(bytes)
    file_write_int(f, length)
    file_write_bytes(f, bytes)

def convert_to_bin(name):
    model = onnx.load(name)
    graph = model.graph
    nodes = graph.node
    inputs = graph.input
    outputs = graph.output
    network = {}
    network["nodes"] = {}
    network["edges"] = {}
    network["inputs"] = []
    network["outputs"] = []
    for input in inputs:
        network["inputs"].append(input.name)
    for output in outputs:
        network["outputs"].append(output.name)
    initializer = graph.initializer

    for i, node in enumerate(nodes):
        node_name = node.name
        input_edge_list = list(node.input)
        output_edge_list = list(node.output)
        node_dict = {"inputs": input_edge_list, "outputs": output_edge_list}
        node_dict["name"] = node_name
        node_dict["type"] = node.op_type
        
        attribute_dict = {}

        for attr in node.attribute:
            if attr.type == onnx.AttributeProto().AttributeType.FLOAT:
                attribute_dict[attr.name] = attr.f
            if attr.type == onnx.AttributeProto().AttributeType.FLOATS:
                attribute_dict[attr.name] = [x for x in attr.floats]
            if attr.type == onnx.AttributeProto().AttributeType.INT:
                attribute_dict[attr.name] = attr.i
            if attr.type == onnx.AttributeProto().AttributeType.INTS:
                attribute_dict[attr.name] = [x for x in attr.ints]
            if attr.type == onnx.AttributeProto().AttributeType.STRING:
                attribute_dict[attr.name] = str(attr.s.decode("UTF-8"))
            if attr.type == onnx.AttributeProto().AttributeType.STRINGS:
                attribute_dict[attr.name] = [str(x.decode("UTF-8")) for x in attr.strings]

        node_dict["attrbiute"] = attribute_dict
        network["nodes"][node_name] = node_dict

    all_tensor_datas = {}

    for i, edge in enumerate(initializer):
        edge_name = edge.name
        np_data = None
        tensor_shape = list(edge.dims)
        edge_dict = {}
        
        edge_dict["shape"] = tensor_shape
        edge_dict["name"] = edge_name
        edge_dict["dtype"] = edge.data_type
        
        #edge_dict["data"] = edge.raw_data
        np_data = np.frombuffer(edge.raw_data, dtype=np.byte)
        
        all_tensor_datas[edge_name] = np_data.tobytes()
        network["edges"][edge_name] = edge_dict
    
    network_data = json.dumps(network)
    network_data_bytes = bytes(network_data, 'utf-8')
    length = len(network_data_bytes)

    save_name = name[:-5] + ".bin"
    with open(save_name, "wb") as f:
        file_write_int(f, length)
        f.write(network_data_bytes)
        for key in all_tensor_datas.keys():
            file_write_str_with_len(f, key)
            data = all_tensor_datas[key]
            file_write_bytes_with_len(f, data)

    return save_name

  
        
