class ConvParams:
    def __init__(self):
        self.pads = None
        self.kernel_shape = None
        self.strides = None
        self.dilations = None
        self.group = 1


class ActivationParams:
    def __init__(self):
        self.type = None
        self.alpha = 1.0
        self.beta = 1.0


class ElementwiseParams:
    def __init__(self):
        self.type = None


class PoolParams:
    def __init__(self):
        self.type = None
        self.pads = []
        self.kernel_shape = []
        self.strides = []
        self.ceil_mode = 0


class FlattenParams:
    def __init__(self):
        self.axis = None
        self.is_inplace = False


class GemmParams:
    def __init__(self):
        self.alpha = None
        self.beta = None
        self.transA = 0
        self.transB = 0


class ResizeParams:
    def __init__(self):
        self.coordinate_transformation_mode = None
        self.cubic_coeff_a = None
        self.mode = None
        self.nearest_mode = None
        self.exclude_outside = None
        self.extrapolation_value = None


class ConcatParams:
    def __init__(self):
        self.axis = None


class TransposeParams:
    def __init__(self):
        self.perm = None


class ReshapeParams:
    def __init__(self):
        self.allowzero = None
        self.shape = None


class CastParams:
    def __init__(self):
        self.in_dtype = "float32"
        self.out_dtype = "float16"


class SplitParams:
    def __init__(self):
        self.axis = None


class SoftmaxParams:
    def __init__(self):
        self.axis = -1


class GatherParams:
    def __init__(self):
        self.axis = None


class SliceParams:
    def __init__(self):
        self.axis = None


class RmsNormParams:
    def __init__(self):
        self.eps = 1e-7


class Qwen2Params:
    def __init__(self):
        self.block_count = 0
        self.context_length = 0
        self.version = 0
        self.tensor_count = 0
        self.kv_count = 0
        self.name = 0
        self.file_type = 0
        self.quantization_version = 0
        self.embedding_length = 0
        self.feed_forward_length = 0
        self.attention_head_count = 0
        self.attention_head_count_kv = 0
        self.attention_layer_norm_rms_epsilon = 0
        self.rope_freq_base = 0
        self.tokenizer_model = 0
        self.tokenizer_pre = 0
        self.tokenizer_tokens = 0
        self.tokenizer_token_type = 0
        self.tokenizer_merges = 0
        self.tokenizer_eos_token_id = 0
        self.tokenizer_padding_token_id = 0
        self.tokenizer_bos_token_id = 0
        self.tokenizer_add_bos_token = 0
        self.tokenizer_scores = 0
        self.tokenizer_chat_template = 0
