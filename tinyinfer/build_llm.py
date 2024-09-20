from read_gguf import get_gguf_data
from build_qwen2 import build_qwen2


def build_llm(keyvalues, weights):
    if keyvalues["general.architecture"] == "qwen2":
        build_qwen2(keyvalues, weights)
    else:
        print("not support ", keyvalues["general.architecture"])
        raise IOError


if __name__ == "__main__":
    name = "../data/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-fp16.gguf"
    name = "data/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-fp16.gguf"
    keyvalues, weights = get_gguf_data(name)
    build_llm(keyvalues, weights)
