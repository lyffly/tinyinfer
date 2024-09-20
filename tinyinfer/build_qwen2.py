import read_gguf


def build_qwen2(keyvalues, weights):
    keyvalues["qwen2.block_count"]

    pass


if __name__ == "__main__":
    name = "../data/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-fp16.gguf"
    name = "data/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-fp16.gguf"
    keyvalues, weights = read_gguf.get_gguf_data(name)
    build_qwen2(keyvalues, weights)
