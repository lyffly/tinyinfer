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
2024-7-30 FPS=574 ??? (FP16), FPS=923 (FP32), 添加cudnn conv2d
