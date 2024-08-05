import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.onnx import export

class Qwen2MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 896
        self.intermediate_size = 4864
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


x = torch.rand(1,512,896)
mlp = Qwen2MLP()
mlp.eval()

torch.onnx.export(
    
        model         = mlp, 
        args          = (x,),
        f             = "mlp.onnx",
        input_names   = ["input0"],
        output_names  = ["output0"],
        opset_version = 12)
