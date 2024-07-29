# tinyinfer

#### 介绍

一个推理框架，目标是最精简的技术实现，开发中。  
更新地址1: https://github.com/lyffly/tinyinfer  
更新地址2: https://gitee.com/yunfeiliu/tinyinfer  
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
- [x] 使用cublas的gemm
- [x] 使用cudnn的conv2d
- [ ] cutlass实现gemm
- [ ] cutlass实现conv2d
- [x] 支持ONNX模型（CV模型）
- [ ] 支持GGUF的模型（大模型）

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

##### 性能

测试 Resnet18 FPS性能为：
2024-7-15 FPS=79 (cpu)  
2024-7-16 FPS=793 (use pytorch backend, 实际TF32, 显卡为：3080Ti)，TensorRT 为1525  
2024-7-23 FPS=1515 (TF32), FPS=1529 (FP32), FPS=4620 (FP16) 使用TensorRT  
2024-7-23 FPS=812 (TF32), FPS=627 (FP32), FPS=735 (FP16), 未融合，使用pytorch  
2024-7-27 FPS=927 (FP16), 添加Relu和Add的CUDA实现  
2024-7-28 FPS=930 (FP16), 添加cublas gemm, 去除h2d和d2h  

#### 愿景

不求速度多快，不求技术多高级，只做技术积累，把一个推理引擎所需要的内容整合完毕。
