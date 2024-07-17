# tinyinfer

#### 介绍

一个推理框架，目标是最精简的技术实现，开发中。
长期计划如下，现在还比较早期，晚上和周末才有时间更新

- [x] 使用Python实现前端接口
- [x] 支持Pytorch backend
- [x] 支持FP32推理
- [ ] 支持FP16推理
- [ ] 支持INT8推理
- [ ] 支持PPQ量化
- [ ] 算子融合
- [ ] 使用C++实现自定义CUDA算子
- [x] 支持ONNX模型（CV模型）
- [ ] 支持GGUF的模型（大模型）


#### 使用Renset18测试

```shell
python3 test_resnet18.py
```

##### 性能

测试 Resnet18 FPS性能为：
2024-7-15 FPS=79 (cpu)  
2024-7-16 FPS=793 (use pytorch backend, FP32, 显卡为：3080Ti)，TensorRT 为1525  

####

不求速度多快，不求技术多高级，只做技术积累，把一个推理引擎所需要的内容整合完毕。
