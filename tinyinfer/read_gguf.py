from gguf.gguf_reader import GGUFReader
from kernels import YTensor, DataType, DataLayout, TensorType
import numpy as np


def gguf_tensor_type_2_ytensor_type(typename):
    if typename == "F32":
        return DataType.float32
    elif typename == "F16":
        return DataType.float16
    else:
        raise IOError


def get_gguf_tensors(reader: GGUFReader):
    ytensors = {}
    for tensor in reader.tensors:
        ytensor = YTensor()
        ytensor.zeros(
            list(tensor.shape),
            gguf_tensor_type_2_ytensor_type(tensor.tensor_type.name),
            DataLayout.nchw,
        )
        ytensor.tensortype = TensorType.constant
        data_np = np.array(tensor.data)
        ytensor.copy_numpy_data(data_np.__array_interface__["data"][0])
        ytensors[tensor.name] = ytensor
        # ytensor.print(10)
    # print(ytensors)
    return ytensors


def get_gguf_key_values(reader: GGUFReader):
    key_values = {}
    for key, field in reader.fields.items():
        value = field.parts[field.data[0]]
        if value.dtype == np.uint8:
            d_str = ""
            for i in range(value.size):
                d = value[i]
                d_str += chr(d)
            key_values[key] = d_str
        elif value.dtype == np.uint32:
            if value.size == 1:
                d = int(value[0])
                key_values[key] = d
            elif value.size > 1:
                d_list = []
                for i in range(value.size):
                    d = int(value[i])
                    d_list.append(d)
                key_values[key] = d_list
        elif value.dtype == np.int32:
            if value.size == 1:
                d = int(value[0])
                key_values[key] = d
            elif value.size > 1:
                d_list = []
                for i in range(value.size):
                    d = int(value[i])
                    d_list.append(d)
                key_values[key] = d_list
        elif value.dtype == np.uint64:
            if value.size == 1:
                d = int(value[0])
                key_values[key] = d
            elif value.size > 1:
                d_list = []
                for i in range(value.size):
                    d = int(value[i])
                    d_list.append(d)
                key_values[key] = d_list
        elif value.dtype == np.float32:
            if value.size == 1:
                d = float(value[0])
                key_values[key] = d
            elif value.size > 1:
                d_list = []
                for i in range(value.size):
                    d = float(value[i])
                    d_list.append(d)
                key_values[key] = d_list
        elif value.dtype == np.bool_:
            if value.size == 1:
                d = bool(value[0])
                key_values[key] = d
            elif value.size > 1:
                d_list = []
                for i in range(value.size):
                    d = bool(value[i])
                    d_list.append(d)
                key_values[key] = d_list
        else:
            print("value.dtype not support :", value.dtype)
            raise IOError
    return key_values


def get_gguf_data(name):
    reader = GGUFReader(name)
    keyvalues = get_gguf_key_values(reader)
    ytensors = get_gguf_tensors(reader)
    del reader
    return keyvalues, ytensors


if __name__ == "__main__":
    name = "../data/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-fp16.gguf"
    name = "data/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-fp16.gguf"
    get_gguf_data(name)
