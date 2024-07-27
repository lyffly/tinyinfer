
class ConvParams :
    def __init__(self):
        self.pads = None
        self.kernel_shape = None
        self.strides = None
        self.dilations = None
        self.group = 1

class ActivationParams :
    def __init__(self):
        self.type = None
        self.alpha = 1.0
        self.beta = 1.0

class ElementwiseParams :
    def __init__(self):
        self.type = None

class PoolParams :
    def __init__(self):
        self.type = None
        self.pads = None
        self.kernel_shape = None
        self.strides = None
        self.ceil_mode = 0

class FlattenParams :
    def __init__(self):
        self.axis = None

class GemmParams :
    def __init__(self):
        self.alpha = None
        self.beta = None
        self.transB = None

class ResizeParams :
    def __init__(self):
        self.coordinate_transformation_mode = None
        self.cubic_coeff_a = None
        self.mode = None
        self.nearest_mode = None

class ConcatParams :
    def __init__(self):
        self.axis = None
        
class TransposeParams :
    def __init__(self):
        self.perm = None

class ReshapeParams :
    def __init__(self):
        self.shape = None

class CastParams :
    def __init__(self):
        self.in_dtype = "float32"
        self.out_dtype = "float16"
