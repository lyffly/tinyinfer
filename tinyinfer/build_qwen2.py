import read_gguf
from params import Qwen2Params, RmsNormParams
from network import Network
import numpy as np
from .nodes import *
from .edges import *


# https://gitee.com/RexHuang936/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py
class Qwen2Model(Network):
    def __init__(self):
        super().__init__()
        self.params = Qwen2Params()

    def add_gemm(self, in_name: str, w_name: str, bias_name=None):
        node_name = w_name[0:-7]
        node = GemmNode()
        node.name = node_name
        out_name = node_name + ".out"
        if bias_name:
            node.input_names = [in_name, w_name, bias_name]
        else:
            node.input_names = [in_name, w_name]
        node.output_names = [out_name]
        self.nodes[node.name] = node
        self.run_orders.append(node.name)
        return out_name

    def add_rms_norm(self, in_name: str, w_name: str):
        node_name = w_name[0:-7]
        node = RmsNormNode()
        node.name = node_name
        out_name = node_name + ".out"
        node.input_names = [in_name, w_name]
        node.output_names = [out_name]
        params = RmsNormParams()
        params.eps = self.params.attention_layer_norm_rms_epsilon
        node.params = params
        self.nodes[node.name] = node
        self.run_orders.append(node.name)
        return out_name

    def add_silu(self, in_name: str):
        node_name = in_name[0:-4] + ".silu"
        node = SiluNode()
        node.name = node_name
        out_name = node_name + ".out"
        node.input_names = [in_name]
        node.output_names = [out_name]
        self.nodes[node.name] = node
        self.run_orders.append(node.name)
        return out_name

    def add_add(self, in_name: str):
        node_name = in_name[0:-4] + ".add"
        node = ElementwiseNode()
        node.name = node_name
        node.type = "Add"
        out_name = node_name + ".out"
        node.input_names = [in_name]
        node.output_names = [out_name]
        self.nodes[node.name] = node
        self.run_orders.append(node.name)
        return out_name

    def add_mul(self, in_name: str, other_name: str):
        node_name = in_name[0:-4] + ".mul"
        node = ElementwiseNode()
        node.type = "Mul"
        node.name = node_name
        out_name = node_name + ".out"
        node.input_names = [in_name, other_name]
        node.output_names = [out_name]
        self.nodes[node.name] = node
        self.run_orders.append(node.name)
        return out_name

    def add_div(self, in_name: str, other_name: str):
        node_name = in_name[0:-4] + ".div"
        node = ElementwiseNode()
        node.type = "Div"
        node.name = node_name
        out_name = node_name + ".out"
        node.input_names = [in_name, other_name]
        node.output_names = [out_name]
        self.nodes[node.name] = node
        self.run_orders.append(node.name)
        return out_name

    def add_softmax(self, in_name: str):
        node_name = in_name[0:-4] + ".softmax"
        node = SoftmaxNode()
        node.name = node_name
        out_name = node_name + ".out"
        node.input_names = [in_name]
        node.output_names = [out_name]
        self.nodes[node.name] = node
        self.run_orders.append(node.name)
        return out_name

    def add_repeat_kv(self, in_name: str):
        node_name = in_name[0:-4] + ".repeatkv"
        node = RepeatKVNode()
        node.name = node_name
        out_name = node_name + ".out"
        node.input_names = [in_name]
        node.output_names = [out_name]
        self.nodes[node.name] = node
        self.run_orders.append(node.name)
        return out_name

    def add_apply_rope(self, in_name: str, freqs_complex_name: str):
        node_name = in_name[0:-4] + ".apply_rope"
        node = ApplyRopeNode()
        node.name = node_name
        out_name = node_name + ".out"
        node.input_names = [in_name, freqs_complex_name]
        node.output_names = [out_name]
        self.nodes[node.name] = node
        self.run_orders.append(node.name)
        return out_name

    def add_ffn(self, layer_id, in_name):
        ffn_norm_weight_name = "blk.{}.ffn_norm.weight".format(layer_id)
        norm_out = self.add_rms_norm(in_name, ffn_norm_weight_name)

        ffn_gate_weight_name = "blk.{}.ffn_gate.weight".format(layer_id)
        gate_out = self.add_gemm(norm_out, ffn_gate_weight_name)

        silu_out = self.add_silu(gate_out)

        ffn_up_weight_name = "blk.{}.ffn_up.weight".format(layer_id)
        up_out = self.add_gemm(norm_out, ffn_up_weight_name)

        self.add_mul()
        mul_out = ElementwiseNode(silu_out, up_out)

        ffn_down_weight_name = "blk.{}.ffn_down.weight".format(layer_id)
        down_out = self.add_gemm(mul_out, ffn_down_weight_name)

        return down_out

    def add_attention(self, layer_id, in_name):
        norm_wts_name = "blk.{}.attn_norm.weight".format(layer_id)
        norm_out = self.add_rms_norm(in_name, norm_wts_name)

        q_wts_name = "blk.{}.attn_q.weight".format(layer_id)
        q_bias_name = "blk.{}.attn_q.bias".format(layer_id)
        q_out = self.add_gemm(norm_out, q_wts_name, q_bias_name)

        k_wts_name = "blk.{}.attn_k.weight".format(layer_id)
        k_bias_name = "blk.{}.attn_k.bias".format(layer_id)
        k_out = self.add_gemm(norm_out, k_wts_name, k_bias_name)

        v_wts_name = "blk.{}.attn_v.weight".format(layer_id)
        v_bias_name = "blk.{}.attn_v.bias".format(layer_id)
        v_out = self.add_gemm(norm_out, v_wts_name, v_bias_name)

        # rope todo
        freqs_complex_name = "TODO"
        q_out = self.add_apply_rope(q_out, freqs_complex_name)
        k_out = self.add_apply_rope(k_out, freqs_complex_name)

        # repeat kv
        k_out = self.add_repeat_kv(k_out)
        v_out = self.add_repeat_kv(v_out)

        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        # attn_weights = attn_weights + causal_mask

        # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # attn_output = torch.matmul(attn_weights, value_states)

        # attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # attn_output = self.o_proj(attn_output)

        out_wts_name = "blk.{}.attn_output.weight".format(layer_id)

        pass

    def add_decode_layer(self, layer_id):
        pass

    def build(self, weights):
        token_wts_name = "token_embd.weight"
        out_norm_wts_name = "output_norm.weight"

        pass


def build_qwen2(keyvalues, weights):
    model = Qwen2Model()
    model.params.block_count = keyvalues["qwen2.block_count"]
    model.params.context_length = keyvalues["qwen2.context_length"]
    model.params.embedding_length = keyvalues["qwen2.embedding_length"]
    model.params.feed_forward_length = keyvalues["qwen2.feed_forward_length"]
    model.params.attention_head_count = keyvalues["qwen2.attention.head_count"]
    model.params.attention_head_count_kv = keyvalues["qwen2.attention.head_count_kv"]
    model.params.attention_layer_norm_rms_epsilon = keyvalues[
        "qwen2.attention.layer_norm_rms_epsilon"
    ]
    model.params.rope_freq_base = keyvalues["qwen2.rope.freq_base"]
    model.params.tokenizer_chat_template = keyvalues["tokenizer.chat_template"]
    model.build(weights)
    return model


if __name__ == "__main__":
    name = "../data/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-fp16.gguf"
    name = "data/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-fp16.gguf"
    keyvalues, weights = read_gguf.get_gguf_data(name)
    build_qwen2(keyvalues, weights)
