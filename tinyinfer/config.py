
class Config:
    def __init__(self):
        self.fp16 = True
        self.fp32 = False
        self.int8 = False
        self.bf16 = False
        
        self.use_gpu = True
        self.use_cpu = False
        
        self.use_cudnn = True
        self.use_torch = False
        self.use_custom_ops = True
        
        self.debug = False
        self.log_verbose = False
