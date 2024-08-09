# tinyinfer

#### 介绍

一个推理框架，目标是最精简的技术实现，开发中。  
更新地址1: <https://github.com/lyffly/tinyinfer>  
更新地址2: <https://gitee.com/yunfeiliu/tinyinfer>  
长期计划如下，现在还比较早期，晚上和周末才有时间更新

- [x] 使用Python实现前端接口
- [x] 支持Pytorch backend
- [x] 支持FP32推理
- [x] 支持FP16推理
- [ ] 支持INT8推理
- [ ] 支持PPQ量化
- [ ] 算子融合
- [x] 使用C++实现自定义CUDA算子
- [x] 使用C++实现tensor类
- [ ] 实现显存池，减少显存使用
- [x] 使用cublas的gemm
- [x] 使用cudnn的conv2d
- [x] 使用cudnn的pooling
- [ ] cutlass实现gemm
- [ ] cutlass实现conv2d
- [x] 支持ONNX模型（CV模型）
- [ ] 支持GGUF的模型（大模型）
- [ ] 引入triton生成后端算子
- [ ] stablehlo,使用mlir生成cubin然后调用

#### 编译

```shell
git clone https://github.com/lyffly/tinyinfer
# or https://gitee.com/yunfeiliu/tinyinfer
cd tinyinfer
git submodule update --init --recursive

# build wheel
python3 setup.py bdist_wheel
# install
pip install dist/*.whl
```

#### 使用Renset18测试

```shell
python3 test_resnet18_fp32.py
python3 test_resnet18_fp16.py
```

#### 愿景

不求速度多快，不求技术多高级，只做技术积累，把一个推理引擎所需要的内容整合完毕。由于在职，部分优化不方便公开。

#### 参考或看过或未来会用到的仓库

1、<https://github.com/OpenPPL/ppl.kernel.cuda>  
2、<https://github.com/OpenPPL/ppl.llm.kernel.cuda>  
3、<https://github.com/alibaba/MNN>  
4、<https://github.com/Tencent/TNN>  
5、<https://github.com/ggerganov/llama.cpp>  
6、<https://github.com/karpathy/llm.c>  
7、<https://github.com/triton-lang/triton>  
8、<https://github.com/NVIDIA/cutlass>  
9、<https://github.com/NVIDIA/TensorRT>  
